"""API endpoints for LangChain agent operations."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from ..agents import AgentType, LangChainAgentManager, MemoryConfig, MemoryType
from ..agents.langchain_manager import AgentConfig, AgentRequest, ModelProvider
from ..config import AgentSettings, get_settings

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


# Request/Response Models
class ChatRequest(BaseModel):
    """Request for conversational agent chat."""

    message: str = Field(..., description="User message")
    session_id: str = Field(default="default", description="Session identifier")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature")
    stream: bool = Field(default=False, description="Enable streaming")


class TaskRequest(BaseModel):
    """Request for task execution with tools."""

    task: str = Field(..., description="Task description")
    session_id: str = Field(default="task", description="Session identifier")
    agent_type: AgentType = Field(
        default=AgentType.ZERO_SHOT_REACT, description="Agent type"
    )
    tools: Optional[List[str]] = Field(
        default=None, description="Specific tools to enable"
    )
    max_iterations: int = Field(default=10, ge=1, le=30, description="Max iterations")
    model: str = Field(default="gpt-4o-mini", description="LLM model")


class MemoryRequest(BaseModel):
    """Request for memory configuration."""

    session_id: str = Field(..., description="Session identifier")
    memory_type: MemoryType = Field(
        default=MemoryType.BUFFER, description="Memory type"
    )
    window_size: int = Field(default=5, ge=1, le=50, description="Window size")


# Dependency for agent manager
_manager: Optional[LangChainAgentManager] = None


async def get_agent_manager(
    settings: AgentSettings = Depends(get_settings),
) -> LangChainAgentManager:
    """Get or create agent manager instance."""
    global _manager

    if _manager is None:
        _manager = LangChainAgentManager(
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            enable_tracing=False,
        )
        await _manager.initialize()
        logger.info("LangChain agent manager initialized")

    return _manager


@router.post("/chat", response_model=Dict[str, Any])
async def chat_with_agent(
    request: ChatRequest,
    manager: LangChainAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """Chat with a conversational agent.

    This endpoint creates a conversational agent with memory that can use tools
    to answer questions and help with tasks.

    Args:
        request: Chat request with message and configuration
        manager: Agent manager dependency

    Returns:
        Response with agent output and metadata
    """
    try:
        # Determine model provider
        provider = (
            ModelProvider.ANTHROPIC
            if "claude" in request.model.lower()
            else ModelProvider.OPENAI
        )

        # Configure agent
        config = AgentConfig(
            agent_type=AgentType.CONVERSATIONAL_REACT,
            model_provider=provider,
            model_name=request.model,
            temperature=request.temperature,
            memory_config=MemoryConfig(memory_type=MemoryType.BUFFER),
        )

        # Handle streaming
        if request.stream:

            async def generate() -> Any:
                agent_request = AgentRequest(
                    message=request.message,
                    session_id=request.session_id,
                    stream=True,
                )
                async for chunk in manager.execute_agent_streaming(
                    agent_request, config
                ):
                    yield chunk

            return StreamingResponse(generate(), media_type="text/plain")

        # Non-streaming response
        agent_request = AgentRequest(
            message=request.message,
            session_id=request.session_id,
            stream=False,
        )

        response = await manager.execute_agent(agent_request, config)

        if response.error:
            raise HTTPException(status_code=500, detail=response.error)

        return {
            "output": response.output,
            "session_id": response.session_id,
            "intermediate_steps": response.intermediate_steps,
        }

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/task", response_model=Dict[str, Any])
async def execute_task(
    request: TaskRequest,
    manager: LangChainAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """Execute a task using an agent with tools.

    This endpoint allows you to execute complex tasks that require multiple
    tool calls and reasoning steps.

    Args:
        request: Task request with description and configuration
        manager: Agent manager dependency

    Returns:
        Response with task result and execution details
    """
    try:
        # Determine model provider
        provider = (
            ModelProvider.ANTHROPIC
            if "claude" in request.model.lower()
            else ModelProvider.OPENAI
        )

        # Configure agent
        config = AgentConfig(
            agent_type=request.agent_type,
            model_provider=provider,
            model_name=request.model,
            max_iterations=request.max_iterations,
            memory_config=MemoryConfig(memory_type=MemoryType.WINDOW, window_size=3),
        )

        # Execute task
        agent_request = AgentRequest(
            message=request.task,
            session_id=request.session_id,
            tool_names=request.tools,
        )

        response = await manager.execute_agent(agent_request, config)

        if response.error:
            raise HTTPException(status_code=500, detail=response.error)

        return {
            "result": response.output,
            "session_id": response.session_id,
            "steps": response.intermediate_steps,
            "step_count": len(response.intermediate_steps),
        }

    except Exception as e:
        logger.error(f"Task execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=Dict[str, Any])
async def list_available_tools(
    manager: LangChainAgentManager = Depends(get_agent_manager),
    category: Optional[str] = Query(None, description="Filter by category"),
) -> Dict[str, Any]:
    """List all available tools for agents.

    Args:
        manager: Agent manager dependency
        category: Optional category filter

    Returns:
        List of available tools with metadata
    """
    try:
        tools = manager.tool_registry.list_tools()

        if category:
            tools = [t for t in tools if t["category"] == category]

        return {
            "total": len(tools),
            "tools": tools,
            "categories": list(set(t["category"] for t in tools)),
        }

    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=Dict[str, Any])
async def list_sessions(
    manager: LangChainAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """List all active agent sessions.

    Args:
        manager: Agent manager dependency

    Returns:
        List of active sessions
    """
    try:
        agents = manager.list_agents()
        memory_sessions = manager.memory_manager.list_sessions()

        return {
            "active_agents": len(agents),
            "agents": agents,
            "memory_sessions": memory_sessions,
        }

    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session_info(
    session_id: str,
    manager: LangChainAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """Get information about a specific session.

    Args:
        session_id: Session identifier
        manager: Agent manager dependency

    Returns:
        Session information
    """
    try:
        memory_info = manager.memory_manager.get_session_info(session_id)

        if not memory_info:
            raise HTTPException(status_code=404, detail="Session not found")

        return memory_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=Dict[str, Any])
async def delete_session(
    session_id: str,
    manager: LangChainAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """Delete a session and clear its memory.

    Args:
        session_id: Session identifier
        manager: Agent manager dependency

    Returns:
        Success status
    """
    try:
        deleted = manager.memory_manager.delete_session(session_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"status": "deleted", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/memory", response_model=Dict[str, Any])
async def configure_session_memory(
    session_id: str,
    request: MemoryRequest,
    manager: LangChainAgentManager = Depends(get_agent_manager),
    settings: AgentSettings = Depends(get_settings),
) -> Dict[str, Any]:
    """Configure memory for a session.

    Args:
        session_id: Session identifier
        request: Memory configuration request
        manager: Agent manager dependency
        settings: Application settings

    Returns:
        Memory configuration status
    """
    try:
        config = MemoryConfig(
            memory_type=request.memory_type,
            window_size=request.window_size,
        )

        manager.memory_manager.config = config
        memory = manager.memory_manager.create_memory(
            session_id=session_id,
            openai_api_key=settings.openai_api_key,
        )

        return {
            "status": "configured",
            "session_id": session_id,
            "memory_type": request.memory_type,
            "config": config.model_dump(),
        }

    except Exception as e:
        logger.error(f"Error configuring memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_agent_stats(
    manager: LangChainAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """Get overall agent system statistics.

    Args:
        manager: Agent manager dependency

    Returns:
        System statistics
    """
    try:
        return manager.get_stats()

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))