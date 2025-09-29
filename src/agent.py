"""Plasma Agent - Core agent orchestration with LLM integration."""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field

import openai
import anthropic
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from .memory import AgentMemory
from .tools import ToolRegistry, ToolResult


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentConfig:
    """Configuration for Plasma Agent."""
    name: str = "PlasmaAgent"
    llm_provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 60
    max_retries: int = 3
    memory_db_path: str = "agent_memory.db"

    # API Keys (should be set via environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Tool configuration
    enable_tools: bool = True
    allowed_shell_commands: Optional[List[str]] = None

    # System prompts
    system_prompt: Optional[str] = None


@dataclass
class Task:
    """Task definition for agent execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    instructions: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Message in conversation."""
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class PlasmaAgent:
    """Main Plasma Agent class with LLM integration and task execution."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Plasma Agent.

        Args:
            config: Agent configuration, uses defaults if None
        """
        self.config = config or AgentConfig()
        self.session_id = str(uuid.uuid4())

        # Initialize components
        self.memory = AgentMemory(self.config.memory_db_path)
        self.tool_registry = ToolRegistry() if self.config.enable_tools else None

        # Initialize LLM clients
        self._openai_client: Optional[openai.AsyncOpenAI] = None
        self._anthropic_client: Optional[anthropic.AsyncAnthropic] = None

        # Task tracking
        self._tasks: Dict[str, Task] = {}
        self._conversation_history: List[Message] = []

        logger.info(f"Initialized PlasmaAgent '{self.config.name}' with session {self.session_id}")

    async def initialize(self) -> None:
        """Initialize agent components."""
        await self.memory.initialize()

        # Initialize LLM clients based on configuration
        if self.config.llm_provider == LLMProvider.OPENAI and self.config.openai_api_key:
            self._openai_client = openai.AsyncOpenAI(api_key=self.config.openai_api_key)

        elif self.config.llm_provider == LLMProvider.ANTHROPIC and self.config.anthropic_api_key:
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)

        logger.info("Agent initialization completed")

    @property
    def llm_client(self) -> Union[openai.AsyncOpenAI, anthropic.AsyncAnthropic]:
        """Get the appropriate LLM client."""
        if self.config.llm_provider == LLMProvider.OPENAI:
            if not self._openai_client:
                raise ValueError("OpenAI client not initialized. Check API key.")
            return self._openai_client
        elif self.config.llm_provider == LLMProvider.ANTHROPIC:
            if not self._anthropic_client:
                raise ValueError("Anthropic client not initialized. Check API key.")
            return self._anthropic_client
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    async def create_task(
        self,
        description: str,
        instructions: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task.

        Args:
            description: Task description
            instructions: Optional detailed instructions
            context: Optional context data

        Returns:
            Task ID
        """
        task = Task(
            description=description,
            instructions=instructions,
            context=context or {}
        )

        self._tasks[task.id] = task

        # Store in memory
        async with self.memory.session_context(self.session_id) as session_memory:
            await session_memory.store(f"task_{task.id}", task.__dict__, "task")

        logger.info(f"Created task {task.id}: {description}")
        return task.id

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        if task_id in self._tasks:
            return self._tasks[task_id]

        # Try to load from memory
        async with self.memory.session_context(self.session_id) as session_memory:
            task_data = await session_memory.retrieve(f"task_{task_id}", "task")
            if task_data:
                task = Task(**task_data)
                self._tasks[task_id] = task
                return task

        return None

    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List tasks, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of tasks
        """
        tasks = list(self._tasks.values())
        if status:
            tasks = [task for task in tasks if task.status == status]
        return tasks

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """Call LLM API with retry logic.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions
            stream: Whether to stream response

        Returns:
            LLM response or async generator for streaming
        """
        try:
            if self.config.llm_provider == LLMProvider.OPENAI:
                return await self._call_openai(messages, tools, stream)
            elif self.config.llm_provider == LLMProvider.ANTHROPIC:
                return await self._call_anthropic(messages, tools, stream)
            else:
                raise ValueError(f"Unsupported provider: {self.config.llm_provider}")

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    async def _call_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """Call OpenAI API."""
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": stream
        }

        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if stream:
            return await self.llm_client.chat.completions.create(**kwargs)
        else:
            response = await self.llm_client.chat.completions.create(**kwargs)
            return response.model_dump()

    async def _call_anthropic(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """Call Anthropic API."""
        # Convert OpenAI format to Anthropic format
        system_message = None
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                converted_messages.append(msg)

        kwargs = {
            "model": self.config.model,
            "messages": converted_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens or 4000,
            "stream": stream
        }

        if system_message:
            kwargs["system"] = system_message

        if tools:
            # Convert tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if tool["type"] == "function":
                    anthropic_tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "input_schema": tool["function"]["parameters"]
                    })
            kwargs["tools"] = anthropic_tools

        if stream:
            return await self.llm_client.messages.create(**kwargs)
        else:
            response = await self.llm_client.messages.create(**kwargs)
            return response.model_dump()

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool call results
        """
        if not self.tool_registry:
            return []

        results = []

        for tool_call in tool_calls:
            try:
                # Handle different formats (OpenAI vs Anthropic)
                if "function" in tool_call:
                    # OpenAI format
                    function_name = tool_call["function"]["name"]
                    parameters = json.loads(tool_call["function"]["arguments"])
                    call_id = tool_call.get("id", str(uuid.uuid4()))
                else:
                    # Anthropic format or direct format
                    function_name = tool_call.get("name", tool_call.get("function_name"))
                    parameters = tool_call.get("input", tool_call.get("parameters", {}))
                    call_id = tool_call.get("id", str(uuid.uuid4()))

                # Execute tool
                result = await self.tool_registry.execute_tool(function_name, parameters)

                # Store result in memory
                async with self.memory.session_context(self.session_id) as session_memory:
                    await session_memory.store(
                        f"tool_result_{call_id}",
                        result.model_dump(),
                        "tool_result"
                    )

                # Format result for LLM
                tool_result = {
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result.model_dump())
                }

                results.append(tool_result)

            except Exception as e:
                logger.error(f"Error executing tool call: {e}")
                error_result = {
                    "tool_call_id": tool_call.get("id", str(uuid.uuid4())),
                    "role": "tool",
                    "name": tool_call.get("function", {}).get("name", "unknown"),
                    "content": json.dumps({"error": str(e)})
                }
                results.append(error_result)

        return results

    async def execute_task(self, task_id: str) -> Task:
        """Execute a task.

        Args:
            task_id: Task identifier

        Returns:
            Updated task object
        """
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if task.status != TaskStatus.PENDING:
            logger.warning(f"Task {task_id} is not pending, current status: {task.status}")
            return task

        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            logger.info(f"Executing task {task_id}: {task.description}")

            # Build conversation context
            messages = []

            # Add system prompt
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})

            # Add task context
            task_prompt = f"Task: {task.description}"
            if task.instructions:
                task_prompt += f"\n\nInstructions: {task.instructions}"
            if task.context:
                task_prompt += f"\n\nContext: {json.dumps(task.context, indent=2)}"

            messages.append({"role": "user", "content": task_prompt})

            # Get available tools
            tools = None
            if self.tool_registry:
                tools = self.tool_registry.create_function_schema()

            # Execute conversation loop
            max_iterations = 10  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Call LLM
                response = await self._call_llm(messages, tools)

                # Extract message content
                if self.config.llm_provider == LLMProvider.OPENAI:
                    assistant_message = response["choices"][0]["message"]
                else:  # Anthropic
                    assistant_message = {
                        "role": "assistant",
                        "content": response["content"][0]["text"] if response["content"] else ""
                    }

                messages.append(assistant_message)

                # Store conversation in memory
                async with self.memory.session_context(self.session_id) as session_memory:
                    await session_memory.store_message(
                        role="assistant",
                        content=assistant_message["content"],
                        tool_calls=assistant_message.get("tool_calls")
                    )

                # Check for tool calls
                tool_calls = assistant_message.get("tool_calls")
                if not tool_calls:
                    # No more tool calls, task is complete
                    break

                # Execute tool calls
                tool_results = await self._execute_tool_calls(tool_calls)
                messages.extend(tool_results)

                # Store tool results
                for tool_result in tool_results:
                    async with self.memory.session_context(self.session_id) as session_memory:
                        await session_memory.store_message(
                            role="tool",
                            content=tool_result["content"]
                        )

            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = assistant_message["content"]

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

        # Update task in memory
        async with self.memory.session_context(self.session_id) as session_memory:
            await session_memory.store(f"task_{task.id}", task.__dict__, "task")

        return task

    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Chat with the agent.

        Args:
            message: User message
            context: Optional context data
            stream: Whether to stream response

        Returns:
            Response string or async generator for streaming
        """
        # Add user message to conversation
        user_msg = Message(role="user", content=message)
        self._conversation_history.append(user_msg)

        # Store in memory
        async with self.memory.session_context(self.session_id) as session_memory:
            await session_memory.store_message("user", message)

        # Build messages for LLM
        messages = []

        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # Add conversation history
        for msg in self._conversation_history:
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name
            messages.append(msg_dict)

        # Get tools
        tools = None
        if self.tool_registry:
            tools = self.tool_registry.create_function_schema()

        if stream:
            return self._stream_chat_response(messages, tools)
        else:
            return await self._complete_chat_response(messages, tools)

    async def _complete_chat_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Get complete chat response."""
        response = await self._call_llm(messages, tools)

        if self.config.llm_provider == LLMProvider.OPENAI:
            assistant_message = response["choices"][0]["message"]
        else:  # Anthropic
            assistant_message = {
                "role": "assistant",
                "content": response["content"][0]["text"] if response["content"] else "",
                "tool_calls": response.get("tool_calls")
            }

        # Handle tool calls
        if assistant_message.get("tool_calls") and self.tool_registry:
            tool_results = await self._execute_tool_calls(assistant_message["tool_calls"])

            # Add assistant message and tool results to conversation
            messages.append(assistant_message)
            messages.extend(tool_results)

            # Get final response
            final_response = await self._call_llm(messages, tools)
            if self.config.llm_provider == LLMProvider.OPENAI:
                final_content = final_response["choices"][0]["message"]["content"]
            else:
                final_content = final_response["content"][0]["text"] if final_response["content"] else ""

            # Store final message
            final_msg = Message(role="assistant", content=final_content)
            self._conversation_history.append(final_msg)

            async with self.memory.session_context(self.session_id) as session_memory:
                await session_memory.store_message("assistant", final_content)

            return final_content
        else:
            # No tool calls, return response directly
            content = assistant_message["content"]
            assistant_msg = Message(role="assistant", content=content)
            self._conversation_history.append(assistant_msg)

            async with self.memory.session_context(self.session_id) as session_memory:
                await session_memory.store_message("assistant", content)

            return content

    async def _stream_chat_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        response_stream = await self._call_llm(messages, tools, stream=True)

        collected_content = ""

        if self.config.llm_provider == LLMProvider.OPENAI:
            async for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    yield content
        else:  # Anthropic
            async for chunk in response_stream:
                if chunk.type == "content_block_delta":
                    content = chunk.delta.text
                    collected_content += content
                    yield content

        # Store complete message
        if collected_content:
            assistant_msg = Message(role="assistant", content=collected_content)
            self._conversation_history.append(assistant_msg)

            async with self.memory.session_context(self.session_id) as session_memory:
                await session_memory.store_message("assistant", collected_content)

    async def get_conversation_history(self, limit: Optional[int] = None) -> List[Message]:
        """Get conversation history.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        if limit:
            return self._conversation_history[-limit:]
        return self._conversation_history.copy()

    async def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()

        async with self.memory.session_context(self.session_id) as session_memory:
            # Clear conversation from memory
            conversations = await session_memory.get_conversation_history()
            if conversations:
                await self.memory.clear_session(self.session_id)

        logger.info("Conversation history cleared")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Statistics dictionary
        """
        stats = await self.memory.get_memory_stats()
        stats.update({
            "session_id": self.session_id,
            "conversation_length": len(self._conversation_history),
            "active_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
            "total_tasks": len(self._tasks)
        })
        return stats

    async def shutdown(self) -> None:
        """Shutdown agent and cleanup resources."""
        logger.info(f"Shutting down PlasmaAgent '{self.config.name}'")

        # Cancel any pending tasks
        for task in self._tasks.values():
            if task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()

        # Close LLM clients if needed
        if hasattr(self._openai_client, 'close'):
            await self._openai_client.close()
        if hasattr(self._anthropic_client, 'close'):
            await self._anthropic_client.close()

        logger.info("Agent shutdown completed")