"""Tests for tool registry and execution system."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools import ToolRegistry, FileReadTool, FileWriteTool, ToolResult


@pytest.mark.unit
class TestToolRegistry:
    """Test cases for ToolRegistry."""

    def test_tool_registry_initialization(self):
        """Test tool registry initialization with default tools."""
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert "file_read" in tools
        assert "file_write" in tools

    @pytest.mark.asyncio
    async def test_execute_tool(self, tool_registry):
        """Test tool execution."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            result = await tool_registry.execute_tool("file_read", {"file_path": temp_path})
            assert result.success is True
            assert result.result == "test content"
        finally:
            Path(temp_path).unlink()

    def test_create_function_schema(self, tool_registry):
        """Test creating OpenAI function schema."""
        schemas = tool_registry.create_function_schema()
        assert len(schemas) > 0
        assert all("type" in s and s["type"] == "function" for s in schemas)
