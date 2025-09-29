"""Tool registry and execution system for Plasma Agent."""

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field
from loguru import logger


class ToolParameter(BaseModel):
    """Tool parameter definition."""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None


class ToolDefinition(BaseModel):
    """Tool definition for LLM function calling."""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Result from tool execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        """Initialize tool with name and description.

        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM function calling.

        Returns:
            ToolDefinition object
        """
        pass


class FileReadTool(BaseTool):
    """Tool for reading file contents."""

    def __init__(self):
        super().__init__(
            name="file_read",
            description="Read contents of a file from the filesystem"
        )

    async def execute(self, file_path: str) -> ToolResult:
        """Read file contents.

        Args:
            file_path: Path to file to read

        Returns:
            ToolResult with file contents or error
        """
        start_time = asyncio.get_event_loop().time()

        try:
            import aiofiles
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()

            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolResult(
                success=True,
                result=content,
                execution_time=execution_time,
                metadata={"file_path": file_path, "size": len(content)}
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error reading file {file_path}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"file_path": file_path}
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to read",
                    required=True
                )
            ]
        )


class FileWriteTool(BaseTool):
    """Tool for writing file contents."""

    def __init__(self):
        super().__init__(
            name="file_write",
            description="Write content to a file on the filesystem"
        )

    async def execute(self, file_path: str, content: str, create_dirs: bool = True) -> ToolResult:
        """Write content to file.

        Args:
            file_path: Path to file to write
            content: Content to write
            create_dirs: Whether to create parent directories

        Returns:
            ToolResult with success status
        """
        start_time = asyncio.get_event_loop().time()

        try:
            import aiofiles
            from pathlib import Path

            path = Path(file_path)
            if create_dirs and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as file:
                await file.write(content)

            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolResult(
                success=True,
                result=f"File written successfully: {file_path}",
                execution_time=execution_time,
                metadata={"file_path": file_path, "size": len(content)}
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error writing file {file_path}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"file_path": file_path}
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to write",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                    required=True
                ),
                ToolParameter(
                    name="create_dirs",
                    type="boolean",
                    description="Whether to create parent directories if they don't exist",
                    required=False,
                    default=True
                )
            ]
        )


class WebSearchTool(BaseTool):
    """Tool for web search using httpx."""

    def __init__(self, search_engine: str = "duckduckgo"):
        super().__init__(
            name="web_search",
            description="Search the web for information"
        )
        self.search_engine = search_engine

    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        """Perform web search.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            ToolResult with search results
        """
        start_time = asyncio.get_event_loop().time()

        try:
            import httpx
            from urllib.parse import quote

            # Simple DuckDuckGo instant answers API (limited but functional)
            encoded_query = quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            # Extract relevant information
            results = []

            # Abstract/answer
            if data.get("Abstract"):
                results.append({
                    "title": data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", ""),
                    "source": data.get("AbstractSource", "")
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100] + "..." if len(topic.get("Text", "")) > 100 else topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "source": "DuckDuckGo"
                    })

            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolResult(
                success=True,
                result=results[:max_results],
                execution_time=execution_time,
                metadata={"query": query, "results_count": len(results)}
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error performing web search: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"query": query}
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query to find information about",
                    required=True
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of search results to return",
                    required=False,
                    default=5
                )
            ]
        )


class ShellCommandTool(BaseTool):
    """Tool for executing shell commands."""

    def __init__(self, allowed_commands: Optional[List[str]] = None):
        super().__init__(
            name="shell_command",
            description="Execute shell commands (restricted for security)"
        )
        # Default safe commands
        self.allowed_commands = allowed_commands or [
            "ls", "pwd", "echo", "cat", "head", "tail", "grep", "find", "wc", "date"
        ]

    async def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """Execute shell command.

        Args:
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            ToolResult with command output
        """
        start_time = asyncio.get_event_loop().time()

        # Security check
        command_name = command.strip().split()[0] if command.strip() else ""
        if command_name not in self.allowed_commands:
            return ToolResult(
                success=False,
                error=f"Command '{command_name}' not allowed. Allowed commands: {', '.join(self.allowed_commands)}",
                execution_time=0,
                metadata={"command": command}
            )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return ToolResult(
                success=process.returncode == 0,
                result={
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "returncode": process.returncode
                },
                execution_time=execution_time,
                metadata={"command": command}
            )

        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout} seconds",
                execution_time=execution_time,
                metadata={"command": command}
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error executing command '{command}': {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"command": command}
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description=f"Shell command to execute. Allowed commands: {', '.join(self.allowed_commands)}",
                    required=True
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds for command execution",
                    required=False,
                    default=30
                )
            ]
        )


class ToolRegistry:
    """Registry for managing and executing tools."""

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._load_default_tools()

    def _load_default_tools(self) -> None:
        """Load default tools into registry."""
        default_tools = [
            FileReadTool(),
            FileWriteTool(),
            WebSearchTool(),
            ShellCommandTool()
        ]

        for tool in default_tools:
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool from the registry.

        Args:
            name: Name of tool to unregister

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Get list of registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get definitions for all registered tools.

        Returns:
            List of tool definitions for LLM function calling
        """
        return [tool.get_definition() for tool in self._tools.values()]

    async def execute_tool(self, name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool with given parameters.

        Args:
            name: Tool name
            parameters: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
                metadata={"available_tools": self.list_tools()}
            )

        logger.info(f"Executing tool: {name} with parameters: {parameters}")

        try:
            result = await tool.execute(**parameters)
            logger.info(f"Tool '{name}' executed successfully in {result.execution_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            return ToolResult(
                success=False,
                error=f"Execution error: {str(e)}",
                metadata={"tool_name": name, "parameters": parameters}
            )

    def create_function_schema(self) -> List[Dict[str, Any]]:
        """Create OpenAI function schema for all tools.

        Returns:
            List of function schemas for OpenAI API
        """
        schemas = []

        for tool in self._tools.values():
            definition = tool.get_definition()

            properties = {}
            required = []

            for param in definition.parameters:
                prop = {
                    "type": param.type,
                    "description": param.description
                }

                if param.enum:
                    prop["enum"] = param.enum

                if param.default is not None:
                    prop["default"] = param.default

                properties[param.name] = prop

                if param.required:
                    required.append(param.name)

            schema = {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }

            schemas.append(schema)

        return schemas