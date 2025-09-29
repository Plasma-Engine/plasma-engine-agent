"""Tests for PlasmaAgent core functionality."""

import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import PlasmaAgent, AgentConfig, TaskStatus


@pytest.mark.unit
class TestPlasmaAgent:
    """Test cases for PlasmaAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent_config):
        """Test agent initialization."""
        agent = PlasmaAgent(agent_config)
        await agent.initialize()

        assert agent.config == agent_config
        assert agent.session_id is not None
        assert agent.memory is not None

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_create_task(self, agent):
        """Test task creation."""
        task_id = await agent.create_task(
            description="Test task",
            instructions="Test instructions"
        )

        assert task_id is not None

        task = await agent.get_task(task_id)
        assert task is not None
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_list_tasks(self, agent):
        """Test listing tasks."""
        task_id1 = await agent.create_task("Task 1")
        task_id2 = await agent.create_task("Task 2")

        all_tasks = await agent.list_tasks()
        assert len(all_tasks) == 2

        pending_tasks = await agent.list_tasks(TaskStatus.PENDING)
        assert len(pending_tasks) == 2
