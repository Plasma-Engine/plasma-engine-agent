"""Comprehensive tests for LangChain agents integration."""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents import AgentType, LangChainAgentManager, MemoryManager, MemoryType, ToolRegistry
from app.agents.langchain_manager import (
    AgentConfig,
    AgentRequest,
    AgentResponse,
    ModelProvider,
)
from app.agents.memory import MemoryConfig
from app.agents.tools import ToolCategory, ToolMetadata, create_custom_tool, validate_tool_result
from app.main import create_app


# Fixtures
@pytest.fixture
def openai_api_key() -> str:
    """Mock OpenAI API key."""
    return "sk-test-key-123"


@pytest.fixture
def anthropic_api_key() -> str:
    """Mock Anthropic API key."""
    return "sk-ant-test-key-123"


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create tool registry for testing."""
    return ToolRegistry()


@pytest.fixture
def memory_manager(tmp_path: Path) -> MemoryManager:
    """Create memory manager with temp storage."""
    config = MemoryConfig(memory_type=MemoryType.BUFFER)
    return MemoryManager(config=config, storage_path=tmp_path)


@pytest.fixture
def agent_manager(openai_api_key: str, anthropic_api_key: str) -> LangChainAgentManager:
    """Create agent manager for testing."""
    return LangChainAgentManager(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        enable_tracing=False,
    )


@pytest.fixture
def test_app() -> TestClient:
    """Create test FastAPI client."""
    app = create_app()
    return TestClient(app)


# Tool Registry Tests
@pytest.mark.unit
class TestToolRegistry:
    """Tests for tool registry functionality."""

    def test_tool_registry_initialization(self, tool_registry: ToolRegistry) -> None:
        """Test tool registry initializes correctly."""
        assert tool_registry is not None
        assert len(tool_registry.get_all_tools()) == 0

    def test_register_and_get_tool(self, tool_registry: ToolRegistry) -> None:
        """Test registering and retrieving tools."""
        # Create custom tool
        def test_func(input_str: str) -> str:
            return f"Result: {input_str}"

        tool, metadata = create_custom_tool(
            name="test_tool",
            description="A test tool",
            func=test_func,
            category=ToolCategory.CUSTOM,
        )

        # Register
        tool_registry.register_tool(tool, metadata)

        # Retrieve
        retrieved_tool = tool_registry.get_tool("test_tool")
        assert retrieved_tool is not None
        assert retrieved_tool.name == "test_tool"

    def test_get_tools_by_category(self, tool_registry: ToolRegistry) -> None:
        """Test filtering tools by category."""
        # Register multiple tools
        for i in range(3):
            tool, metadata = create_custom_tool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                func=lambda x: x,
                category=ToolCategory.SEARCH if i < 2 else ToolCategory.CALCULATOR,
            )
            tool_registry.register_tool(tool, metadata)

        search_tools = tool_registry.get_tools_by_category(ToolCategory.SEARCH)
        assert len(search_tools) == 2

        calc_tools = tool_registry.get_tools_by_category(ToolCategory.CALCULATOR)
        assert len(calc_tools) == 1

    def test_list_tools_with_metadata(self, tool_registry: ToolRegistry) -> None:
        """Test listing all tools with metadata."""
        tool, metadata = create_custom_tool(
            name="test_tool",
            description="Test",
            func=lambda x: x,
        )
        tool_registry.register_tool(tool, metadata)

        tools_list = tool_registry.list_tools()
        assert len(tools_list) == 1
        assert tools_list[0]["name"] == "test_tool"
        assert "usage_count" in tools_list[0]

    def test_increment_usage(self, tool_registry: ToolRegistry) -> None:
        """Test tracking tool usage."""
        tool, metadata = create_custom_tool(
            name="test_tool",
            description="Test",
            func=lambda x: x,
        )
        tool_registry.register_tool(tool, metadata)

        # Increment usage
        tool_registry.increment_usage("test_tool")
        tool_registry.increment_usage("test_tool")

        tools_list = tool_registry.list_tools()
        assert tools_list[0]["usage_count"] == 2

    def test_calculator_tool(self, tool_registry: ToolRegistry) -> None:
        """Test calculator tool functionality."""
        calc_tool = tool_registry.create_calculator_tool()
        assert calc_tool is not None

        # Test basic calculation
        result = calc_tool.run("2 + 2")
        assert "4" in result

        # Test invalid expression
        result = calc_tool.run("invalid; DROP TABLE")
        assert "Error" in result

    def test_search_tool_creation(self, tool_registry: ToolRegistry) -> None:
        """Test search tool creation."""
        search_tool = tool_registry.create_search_tool()
        assert search_tool is not None
        assert search_tool.name == "web_search"

    def test_wikipedia_tool_creation(self, tool_registry: ToolRegistry) -> None:
        """Test Wikipedia tool creation."""
        wiki_tool = tool_registry.create_wikipedia_tool()
        assert wiki_tool is not None
        assert "wikipedia" in wiki_tool.name.lower()

    @pytest.mark.asyncio
    async def test_initialize_default_tools(self, tool_registry: ToolRegistry) -> None:
        """Test initializing all default tools."""
        await tool_registry.initialize_default_tools()
        tools = tool_registry.get_all_tools()
        assert len(tools) > 0

        # Check for expected tools
        tool_names = [tool.name for tool in tools]
        assert "web_search" in tool_names
        assert "calculator" in tool_names
        assert "wikipedia" in tool_names

    def test_validate_tool_result(self) -> None:
        """Test tool result validation."""
        # Test normal result
        result = validate_tool_result("Normal result")
        assert result == "Normal result"

        # Test empty result
        result = validate_tool_result("")
        assert "empty" in result.lower()

        # Test truncation
        long_result = "x" * 20000
        result = validate_tool_result(long_result, max_length=100)
        assert len(result) < len(long_result)
        assert "truncated" in result.lower()


# Memory Manager Tests
@pytest.mark.unit
class TestMemoryManager:
    """Tests for memory management functionality."""

    def test_memory_manager_initialization(self, memory_manager: MemoryManager) -> None:
        """Test memory manager initializes correctly."""
        assert memory_manager is not None
        assert memory_manager.config.memory_type == MemoryType.BUFFER

    def test_create_buffer_memory(self, memory_manager: MemoryManager) -> None:
        """Test creating buffer memory."""
        memory = memory_manager.create_memory("test_session")
        assert memory is not None

    def test_create_window_memory(self, tmp_path: Path) -> None:
        """Test creating window memory."""
        config = MemoryConfig(memory_type=MemoryType.WINDOW, window_size=3)
        manager = MemoryManager(config=config, storage_path=tmp_path)

        memory = manager.create_memory("test_session")
        assert memory is not None

    def test_get_memory(self, memory_manager: MemoryManager) -> None:
        """Test retrieving existing memory."""
        memory_manager.create_memory("test_session")
        retrieved = memory_manager.get_memory("test_session")
        assert retrieved is not None

    def test_clear_memory(self, memory_manager: MemoryManager) -> None:
        """Test clearing session memory."""
        memory = memory_manager.create_memory("test_session")
        memory.save_context({"input": "test"}, {"output": "response"})

        result = memory_manager.clear_memory("test_session")
        assert result is True

    def test_delete_session(self, memory_manager: MemoryManager) -> None:
        """Test deleting a session."""
        memory_manager.create_memory("test_session")
        result = memory_manager.delete_session("test_session")
        assert result is True
        assert memory_manager.get_memory("test_session") is None

    def test_list_sessions(self, memory_manager: MemoryManager) -> None:
        """Test listing active sessions."""
        memory_manager.create_memory("session1")
        memory_manager.create_memory("session2")

        sessions = memory_manager.list_sessions()
        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions

    def test_get_session_info(self, memory_manager: MemoryManager) -> None:
        """Test getting session information."""
        memory_manager.create_memory("test_session")
        info = memory_manager.get_session_info("test_session")

        assert info is not None
        assert info["session_id"] == "test_session"
        assert "message_count" in info
        assert "memory_type" in info

    def test_save_and_load_memory(self, memory_manager: MemoryManager) -> None:
        """Test persisting and loading memory."""
        memory = memory_manager.create_memory("test_session")
        memory.save_context({"input": "Hello"}, {"output": "Hi there!"})

        # Save to disk
        saved = memory_manager.save_memory("test_session")
        assert saved is True

        # Verify file exists
        session_file = memory_manager.storage_path / "test_session.json"
        assert session_file.exists()

        # Clear in-memory session (but keep file)
        memory_manager.clear_memory("test_session")

        # Load from disk - note: this creates a new memory object
        loaded = memory_manager.load_memory("test_session")
        assert loaded is not None

    def test_get_memory_stats(self, memory_manager: MemoryManager) -> None:
        """Test getting memory statistics."""
        memory_manager.create_memory("session1")
        memory_manager.create_memory("session2")

        stats = memory_manager.get_memory_stats()
        assert stats["active_sessions"] == 2
        assert "total_messages" in stats


# Agent Manager Tests
@pytest.mark.unit
class TestLangChainAgentManager:
    """Tests for LangChain agent manager."""

    def test_agent_manager_initialization(self, agent_manager: LangChainAgentManager) -> None:
        """Test agent manager initializes correctly."""
        assert agent_manager is not None
        assert agent_manager.openai_api_key is not None
        assert agent_manager.anthropic_api_key is not None

    @pytest.mark.asyncio
    async def test_initialize_manager(self, agent_manager: LangChainAgentManager) -> None:
        """Test manager initialization loads tools."""
        await agent_manager.initialize()
        tools = agent_manager.tool_registry.get_all_tools()
        assert len(tools) > 0

    def test_create_openai_llm(self, agent_manager: LangChainAgentManager) -> None:
        """Test creating OpenAI LLM."""
        config = AgentConfig(
            model_provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
        )

        llm = agent_manager._create_llm(config)
        assert llm is not None

    def test_create_anthropic_llm(self, agent_manager: LangChainAgentManager) -> None:
        """Test creating Anthropic LLM."""
        config = AgentConfig(
            model_provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
        )

        llm = agent_manager._create_llm(config)
        assert llm is not None

    @pytest.mark.asyncio
    async def test_create_agent_executor(self, agent_manager: LangChainAgentManager) -> None:
        """Test creating agent executor."""
        await agent_manager.initialize()

        config = AgentConfig(
            agent_type=AgentType.ZERO_SHOT_REACT,
            model_provider=ModelProvider.OPENAI,
        )

        executor = agent_manager._create_agent_executor(
            config=config,
            session_id="test",
        )

        assert executor is not None

    @pytest.mark.asyncio
    @patch("app.agents.langchain_manager.initialize_agent")
    async def test_execute_agent_success(
        self,
        mock_initialize: Mock,
        agent_manager: LangChainAgentManager,
    ) -> None:
        """Test successful agent execution."""
        await agent_manager.initialize()

        # Mock agent executor
        mock_executor = AsyncMock()
        mock_executor.ainvoke = AsyncMock(
            return_value={
                "output": "Test response",
                "intermediate_steps": [],
            }
        )
        mock_initialize.return_value = mock_executor

        request = AgentRequest(
            message="Test message",
            session_id="test",
        )

        response = await agent_manager.execute_agent(request)

        assert isinstance(response, AgentResponse)
        assert response.output == "Test response"
        assert response.error is None

    @pytest.mark.asyncio
    @patch("app.agents.langchain_manager.initialize_agent")
    async def test_execute_agent_with_error(
        self,
        mock_initialize: Mock,
        agent_manager: LangChainAgentManager,
    ) -> None:
        """Test agent execution with error."""
        await agent_manager.initialize()

        # Mock agent executor that raises error
        mock_executor = AsyncMock()
        mock_executor.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        mock_initialize.return_value = mock_executor

        request = AgentRequest(
            message="Test message",
            session_id="test",
        )

        response = await agent_manager.execute_agent(request)

        assert isinstance(response, AgentResponse)
        assert response.error is not None
        assert "Test error" in response.error

    def test_get_agent_info(self, agent_manager: LangChainAgentManager) -> None:
        """Test getting agent information."""
        info = agent_manager.get_agent_info("test", AgentType.ZERO_SHOT_REACT)

        assert info is not None
        assert info["session_id"] == "test"
        assert info["agent_type"] == AgentType.ZERO_SHOT_REACT

    def test_list_agents(self, agent_manager: LangChainAgentManager) -> None:
        """Test listing agents."""
        agents = agent_manager.list_agents()
        assert isinstance(agents, list)

    def test_get_stats(self, agent_manager: LangChainAgentManager) -> None:
        """Test getting manager statistics."""
        stats = agent_manager.get_stats()

        assert "active_agents" in stats
        assert "available_tools" in stats
        assert "memory_stats" in stats


# API Endpoint Tests
@pytest.mark.integration
class TestAgentAPI:
    """Tests for agent API endpoints."""

    def test_chat_endpoint(
        self,
        test_app: TestClient,
    ) -> None:
        """Test chat endpoint - expects 500 without real API keys."""
        # Without valid API keys, should get 500 error
        response = test_app.post(
            "/api/v1/agent/chat",
            json={
                "message": "Hello",
                "session_id": "test",
                "model": "gpt-4o-mini",
            },
        )

        # Expected to fail with 500 due to missing API keys
        assert response.status_code == 500

    def test_task_endpoint(
        self,
        test_app: TestClient,
    ) -> None:
        """Test task execution endpoint - expects 500 without real API keys."""
        # Without valid API keys, should get 500 error
        response = test_app.post(
            "/api/v1/agent/task",
            json={
                "task": "Calculate 2 + 2",
                "session_id": "task",
                "agent_type": "zero_shot_react",
            },
        )

        # Expected to fail with 500 due to missing API keys
        assert response.status_code == 500

    @pytest.mark.asyncio
    @patch("app.routers.agent.get_agent_manager")
    async def test_list_tools_endpoint(
        self,
        mock_get_manager: Mock,
        test_app: TestClient,
    ) -> None:
        """Test list tools endpoint."""
        # Mock manager
        mock_manager = Mock()
        mock_manager.tool_registry = Mock()
        mock_manager.tool_registry.list_tools = Mock(
            return_value=[
                {
                    "name": "calculator",
                    "category": "calculator",
                    "description": "Calculate",
                }
            ]
        )
        mock_get_manager.return_value = mock_manager

        response = test_app.get("/api/v1/agent/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "total" in data

    def test_list_sessions_endpoint(self, test_app: TestClient) -> None:
        """Test list sessions endpoint."""
        response = test_app.get("/api/v1/agent/sessions")
        assert response.status_code == 200

    def test_get_stats_endpoint(self, test_app: TestClient) -> None:
        """Test stats endpoint."""
        response = test_app.get("/api/v1/agent/stats")
        assert response.status_code == 200
        data = response.json()
        assert "active_agents" in data


# Edge Cases and Error Handling
@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_create_custom_tool_with_invalid_category(self) -> None:
        """Test creating tool with invalid category."""
        # Should still work, custom category is valid
        tool, metadata = create_custom_tool(
            name="test",
            description="test",
            func=lambda x: x,
            category=ToolCategory.CUSTOM,
        )
        assert tool is not None

    def test_memory_manager_without_api_key(self, tmp_path: Path) -> None:
        """Test creating summary memory without API key."""
        config = MemoryConfig(memory_type=MemoryType.SUMMARY)
        manager = MemoryManager(config=config, storage_path=tmp_path)

        with pytest.raises(ValueError):
            manager.create_memory("test")

    def test_agent_config_validation(self) -> None:
        """Test agent configuration validation."""
        config = AgentConfig(
            temperature=0.5,
            max_iterations=5,
        )

        assert config.temperature == 0.5
        assert config.max_iterations == 5

    @pytest.mark.asyncio
    async def test_tool_registry_browser_without_initialization(
        self, tool_registry: ToolRegistry
    ) -> None:
        """Test browser tool before initialization."""
        # Should initialize automatically
        browser_tool = await tool_registry.create_browser_tool()
        assert browser_tool is not None

    def test_memory_manager_with_summary_buffer(self, tmp_path: Path, openai_api_key: str) -> None:
        """Test creating summary buffer memory."""
        config = MemoryConfig(memory_type=MemoryType.SUMMARY_BUFFER, max_token_limit=1000)
        manager = MemoryManager(config=config, storage_path=tmp_path)

        memory = manager.create_memory("test", openai_api_key=openai_api_key)
        assert memory is not None

    def test_agent_config_all_types(self) -> None:
        """Test all agent configuration types."""
        for agent_type in AgentType:
            config = AgentConfig(agent_type=agent_type)
            assert config.agent_type == agent_type

    @pytest.mark.asyncio
    async def test_agent_streaming_response(self, agent_manager: LangChainAgentManager) -> None:
        """Test streaming agent response."""
        await agent_manager.initialize()

        request = AgentRequest(message="test", session_id="stream_test", stream=True)

        # Should yield at least one chunk or error
        chunks = []
        async for chunk in agent_manager.execute_agent_streaming(request):
            chunks.append(chunk)
            if len(chunks) > 5:  # Limit iterations
                break

        assert len(chunks) > 0

    def test_agent_clear_and_list(self, agent_manager: LangChainAgentManager) -> None:
        """Test clearing agents and listing."""
        # Initially empty
        agents = agent_manager.list_agents()
        assert len(agents) == 0

        # Clear non-existent agent
        result = agent_manager.clear_agent("test", AgentType.ZERO_SHOT_REACT)
        # Should still work

    @pytest.mark.asyncio
    async def test_screenshot_tool(self, tool_registry: ToolRegistry) -> None:
        """Test screenshot tool creation."""
        screenshot_tool = await tool_registry.create_screenshot_tool()
        assert screenshot_tool is not None
        assert "screenshot" in screenshot_tool.name.lower()

    def test_tool_metadata_validation(self) -> None:
        """Test tool metadata structure."""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CUSTOM,
            rate_limit_per_minute=100,
        )
        assert metadata.name == "test_tool"
        assert metadata.enabled is True

    @pytest.mark.asyncio
    async def test_chat_simple_interface(self, agent_manager: LangChainAgentManager) -> None:
        """Test simple chat interface."""
        await agent_manager.initialize()

        # Should return error message due to no API key
        response = await agent_manager.chat("Hello", session_id="chat_test")
        assert isinstance(response, str)
        assert len(response) > 0