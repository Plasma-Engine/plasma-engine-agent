"""Local test configuration for the Agent service."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# -- Path management ------------------------------------------------------
# The service uses a classic ``app/`` package layout instead of the ``src/``
# layout that editable installs automatically expose on ``sys.path``. When
# pytest spins up in a clean environment (for example inside GitHub Actions)
# the repository root is *not* present on ``sys.path`` which makes
# ``import app`` fail before our fixtures run. Adding the service root as the
# very first entry keeps local developer workflows unchanged while ensuring
# automation environments resolve ``app`` reliably.
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import AgentSettings
from app.main import create_app
from src.agent import AgentConfig, PlasmaAgent
from src.memory import AgentMemory
from src.tools import ToolRegistry


@pytest.fixture
def test_settings():
    """Provide test-specific settings."""
    return AgentSettings(
        app_name="plasma-engine-agent-test",
        cors_origins=["http://localhost:3000", "http://localhost:8000"],
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key"
    )


@pytest.fixture
def app(test_settings):
    """Create FastAPI app with test settings."""
    return create_app(test_settings)


@pytest.fixture
def client(app):
    """Provide TestClient for the agent service."""
    return TestClient(app)


@pytest.fixture
def temp_db(tmp_path) -> str:
    """Create an isolated SQLite file for agent memory tests.

    Using ``tmp_path`` keeps the database on disk (matching production usage)
    while ensuring every test run gets a clean file system sandbox.
    """

    return str(tmp_path / "agent-memory.sqlite")


@pytest.fixture
def agent_config(temp_db: str) -> AgentConfig:
    """Construct an :class:`AgentConfig` tailored for unit tests.

    We point ``memory_db_path`` at the temporary database and provide stub API
    keys so initialization paths that rely on configuration can execute without
    external secrets.
    """

    return AgentConfig(
        name="TestAgent",
        memory_db_path=temp_db,
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
    )


@pytest_asyncio.fixture
async def agent(agent_config: AgentConfig) -> AsyncGenerator[PlasmaAgent, None]:
    """Provide an initialized :class:`PlasmaAgent` and tear it down safely."""

    instance = PlasmaAgent(agent_config)
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest_asyncio.fixture
async def memory(temp_db: str) -> AsyncGenerator[AgentMemory, None]:
    """Yield an initialized :class:`AgentMemory` instance backed by ``temp_db``."""

    agent_memory = AgentMemory(temp_db)
    await agent_memory.initialize()
    yield agent_memory


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Return a tool registry populated with the default file utilities."""

    return ToolRegistry()


@pytest.fixture
def mock_openai_client():
    """Provide a mock OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content="This is a mock AI response",
                role="assistant"
            ),
            finish_reason="stop"
        )
    ]
    mock_response.usage = Mock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Provide a mock Anthropic client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a mock Claude response", type="text")]
    mock_response.usage = Mock(input_tokens=15, output_tokens=25)
    mock_response.role = "assistant"
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_agent_task():
    """Provide mock agent task data."""
    return {
        "id": "task-123",
        "agent_type": "research",
        "prompt": "Analyze the latest AI developments",
        "parameters": {
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "status": "pending"
    }


@pytest.fixture
def mock_agent_response():
    """Provide mock agent response data."""
    return {
        "task_id": "task-123",
        "agent_type": "research",
        "result": {
            "content": "AI development analysis results...",
            "metadata": {
                "tokens_used": 500,
                "confidence": 0.89,
                "sources": ["paper1.pdf", "article2.html"]
            }
        },
        "status": "completed",
        "execution_time": 2.5
    }