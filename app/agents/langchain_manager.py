"""LangChain agent factory and orchestration manager."""

import asyncio
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain.agents import AgentExecutor, AgentType as LCAgentType, create_react_agent
from langchain.agents import initialize_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .memory import MemoryConfig, MemoryManager, MemoryType
from .tools import ToolRegistry, validate_tool_result


class AgentType(str, Enum):
    """Types of LangChain agents supported."""

    ZERO_SHOT_REACT = "zero_shot_react"  # Zero-shot ReAct agent
    CONVERSATIONAL_REACT = "conversational_react"  # Conversational ReAct agent
    STRUCTURED_CHAT = "structured_chat"  # Structured chat agent
    OPENAI_FUNCTIONS = "openai_functions"  # OpenAI functions agent


class ModelProvider(str, Enum):
    """LLM providers supported."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AgentConfig(BaseModel):
    """Configuration for agent creation."""

    agent_type: AgentType = AgentType.CONVERSATIONAL_REACT
    model_provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_iterations: int = 10
    max_execution_time: Optional[float] = 60.0
    verbose: bool = True
    handle_parsing_errors: bool = True
    return_intermediate_steps: bool = True
    memory_config: Optional[MemoryConfig] = None


class AgentRequest(BaseModel):
    """Request model for agent execution."""

    message: str = Field(..., description="User message or task")
    session_id: str = Field(default="default", description="Session identifier")
    tool_names: Optional[List[str]] = Field(
        default=None, description="Specific tools to use"
    )
    stream: bool = Field(default=False, description="Enable streaming response")


class AgentResponse(BaseModel):
    """Response model from agent execution."""

    output: str
    intermediate_steps: List[Dict[str, Any]] = []
    session_id: str
    total_tokens: Optional[int] = None
    error: Optional[str] = None


class LangChainAgentManager:
    """Manager for creating and orchestrating LangChain agents."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        enable_tracing: bool = False,
    ) -> None:
        """Initialize LangChain agent manager.

        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            enable_tracing: Enable LangSmith tracing
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.enable_tracing = enable_tracing
        self.tool_registry = ToolRegistry()
        self.memory_manager = MemoryManager()
        self._agents: Dict[str, AgentExecutor] = {}
        self._rate_limits: Dict[str, int] = {}

    async def initialize(self) -> None:
        """Initialize the manager and load default tools."""
        logger.info("Initializing LangChain agent manager")
        await self.tool_registry.initialize_default_tools()
        logger.info(
            f"Loaded {len(self.tool_registry.get_all_tools())} tools into registry"
        )

    def _create_llm(
        self, config: AgentConfig
    ) -> Union[ChatOpenAI, ChatAnthropic]:
        """Create LLM instance based on configuration.

        Args:
            config: Agent configuration

        Returns:
            LLM instance
        """
        if config.model_provider == ModelProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                openai_api_key=self.openai_api_key,
                streaming=True,
            )

        elif config.model_provider == ModelProvider.ANTHROPIC:
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")

            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                anthropic_api_key=self.anthropic_api_key,
                streaming=True,
            )

        else:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")

    def _create_agent_executor(
        self,
        config: AgentConfig,
        session_id: str,
        tool_names: Optional[List[str]] = None,
    ) -> AgentExecutor:
        """Create an agent executor with specified configuration.

        Args:
            config: Agent configuration
            session_id: Session identifier
            tool_names: Specific tools to use (None for all)

        Returns:
            Configured agent executor
        """
        # Get LLM
        llm = self._create_llm(config)

        # Get tools
        if tool_names:
            tools = [
                tool
                for tool in self.tool_registry.get_all_tools()
                if tool.name in tool_names
            ]
        else:
            tools = self.tool_registry.get_all_tools()

        if not tools:
            raise ValueError("No tools available for agent")

        # Get or create memory
        memory_config = config.memory_config or MemoryConfig()
        memory = self.memory_manager.create_memory(
            session_id=session_id,
            openai_api_key=self.openai_api_key,
        )

        # Create agent based on type
        if config.agent_type == AgentType.ZERO_SHOT_REACT:
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=LCAgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=config.verbose,
                handle_parsing_errors=config.handle_parsing_errors,
                max_iterations=config.max_iterations,
                max_execution_time=config.max_execution_time,
                return_intermediate_steps=config.return_intermediate_steps,
            )

        elif config.agent_type == AgentType.CONVERSATIONAL_REACT:
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=LCAgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=config.verbose,
                handle_parsing_errors=config.handle_parsing_errors,
                max_iterations=config.max_iterations,
                max_execution_time=config.max_execution_time,
                return_intermediate_steps=config.return_intermediate_steps,
            )

        elif config.agent_type == AgentType.STRUCTURED_CHAT:
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=LCAgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                memory=memory,
                verbose=config.verbose,
                handle_parsing_errors=config.handle_parsing_errors,
                max_iterations=config.max_iterations,
                max_execution_time=config.max_execution_time,
                return_intermediate_steps=config.return_intermediate_steps,
            )

        elif config.agent_type == AgentType.OPENAI_FUNCTIONS:
            if config.model_provider != ModelProvider.OPENAI:
                raise ValueError("OpenAI Functions agent requires OpenAI provider")

            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=LCAgentType.OPENAI_FUNCTIONS,
                memory=memory,
                verbose=config.verbose,
                handle_parsing_errors=config.handle_parsing_errors,
                max_iterations=config.max_iterations,
                max_execution_time=config.max_execution_time,
                return_intermediate_steps=config.return_intermediate_steps,
            )

        else:
            raise ValueError(f"Unsupported agent type: {config.agent_type}")

        return agent

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def execute_agent(
        self, request: AgentRequest, config: Optional[AgentConfig] = None
    ) -> AgentResponse:
        """Execute agent with retry logic and error handling.

        Args:
            request: Agent execution request
            config: Agent configuration (uses defaults if None)

        Returns:
            Agent response
        """
        config = config or AgentConfig()

        try:
            # Create or get agent executor
            agent_key = f"{request.session_id}_{config.agent_type}"
            if agent_key not in self._agents:
                self._agents[agent_key] = self._create_agent_executor(
                    config=config,
                    session_id=request.session_id,
                    tool_names=request.tool_names,
                )

            agent = self._agents[agent_key]

            # Execute agent
            logger.info(
                f"Executing agent {config.agent_type} for session {request.session_id}"
            )

            result = await agent.ainvoke({"input": request.message})

            # Extract output and steps
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Format intermediate steps
            formatted_steps = []
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) >= 2:
                    action, observation = step[0], step[1]
                    formatted_steps.append(
                        {
                            "tool": getattr(action, "tool", "unknown"),
                            "tool_input": getattr(action, "tool_input", ""),
                            "observation": validate_tool_result(str(observation)),
                        }
                    )

            return AgentResponse(
                output=output,
                intermediate_steps=formatted_steps,
                session_id=request.session_id,
            )

        except Exception as e:
            logger.error(f"Agent execution error: {str(e)}")
            return AgentResponse(
                output="",
                session_id=request.session_id,
                error=str(e),
            )

    async def execute_agent_streaming(
        self, request: AgentRequest, config: Optional[AgentConfig] = None
    ) -> AsyncIterator[str]:
        """Execute agent with streaming response.

        Args:
            request: Agent execution request
            config: Agent configuration

        Yields:
            Streaming response chunks
        """
        config = config or AgentConfig()

        try:
            # Create agent executor
            agent = self._create_agent_executor(
                config=config,
                session_id=request.session_id,
                tool_names=request.tool_names,
            )

            # Stream execution
            async for chunk in agent.astream({"input": request.message}):
                if isinstance(chunk, dict):
                    if "output" in chunk:
                        yield chunk["output"]
                    elif "intermediate_step" in chunk:
                        step = chunk["intermediate_step"]
                        if isinstance(step, tuple) and len(step) >= 2:
                            yield f"\n[Tool: {step[0].tool}]\n"
                else:
                    yield str(chunk)

        except Exception as e:
            logger.error(f"Streaming execution error: {str(e)}")
            yield f"\n[Error: {str(e)}]\n"

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        config: Optional[AgentConfig] = None,
    ) -> str:
        """Simple chat interface for conversational agents.

        Args:
            message: User message
            session_id: Session identifier
            config: Agent configuration

        Returns:
            Agent response
        """
        config = config or AgentConfig(agent_type=AgentType.CONVERSATIONAL_REACT)

        request = AgentRequest(message=message, session_id=session_id)
        response = await self.execute_agent(request, config)

        return response.output if not response.error else f"Error: {response.error}"

    def get_agent_info(self, session_id: str, agent_type: AgentType) -> Dict[str, Any]:
        """Get information about an agent.

        Args:
            session_id: Session identifier
            agent_type: Agent type

        Returns:
            Agent information
        """
        agent_key = f"{session_id}_{agent_type}"
        memory_info = self.memory_manager.get_session_info(session_id)

        return {
            "session_id": session_id,
            "agent_type": agent_type,
            "exists": agent_key in self._agents,
            "memory": memory_info,
            "available_tools": [tool.name for tool in self.tool_registry.get_all_tools()],
        }

    def clear_agent(self, session_id: str, agent_type: AgentType) -> bool:
        """Clear an agent and its memory.

        Args:
            session_id: Session identifier
            agent_type: Agent type

        Returns:
            True if cleared, False otherwise
        """
        agent_key = f"{session_id}_{agent_type}"
        if agent_key in self._agents:
            del self._agents[agent_key]

        return self.memory_manager.clear_memory(session_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all active agents.

        Returns:
            List of agent information
        """
        agents = []
        for agent_key in self._agents.keys():
            session_id, agent_type = agent_key.rsplit("_", 1)
            agents.append(
                {
                    "session_id": session_id,
                    "agent_type": agent_type,
                }
            )
        return agents

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "active_agents": len(self._agents),
            "available_tools": len(self.tool_registry.get_all_tools()),
            "memory_stats": self.memory_manager.get_memory_stats(),
            "tool_list": self.tool_registry.list_tools(),
        }