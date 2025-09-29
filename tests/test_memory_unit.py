"""Tests for agent memory management system."""

import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory import AgentMemory


@pytest.mark.unit
class TestAgentMemory:
    """Test cases for AgentMemory class."""

    @pytest.mark.asyncio
    async def test_memory_initialization(self, temp_db):
        """Test memory database initialization."""
        memory = AgentMemory(temp_db)
        await memory.initialize()
        assert Path(temp_db).exists()

    @pytest.mark.asyncio
    async def test_store_and_retrieve_memory(self, memory):
        """Test storing and retrieving memory entries."""
        session_id = "test_session"
        key = "test_key"
        value = {"data": "test_value", "number": 42}

        await memory.store(session_id, key, value)
        retrieved = await memory.retrieve(session_id, key)
        assert retrieved == value

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_memory(self, memory):
        """Test retrieving non-existent memory returns None."""
        result = await memory.retrieve("nonexistent_session", "nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_memory_stats(self, memory):
        """Test memory usage statistics."""
        session_id = "test_session"
        stats = await memory.get_memory_stats()
        initial_count = stats["memories_count"]

        await memory.store(session_id, "key1", "value1")
        stats = await memory.get_memory_stats()
        assert stats["memories_count"] == initial_count + 1
