"""Workflow API endpoints for workflow execution and management."""

from typing import List, Optional
from uuid import uuid4

import yaml
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from ..config import get_settings
from ..models.workflow import (
    WorkflowCreateRequest,
    WorkflowDefinition,
    WorkflowExecuteRequest,
    WorkflowState,
    WorkflowStatusResponse,
)
from ..workflows.engine import WorkflowEngine
from ..workflows.state_manager import WorkflowStateManager

router = APIRouter(prefix="/api/v1/agent/workflows", tags=["workflows"])

# Global instances (initialized on first request)
_state_manager: Optional[WorkflowStateManager] = None
_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get or create workflow engine instance."""
    global _state_manager, _engine

    if not _state_manager:
        settings = get_settings()
        redis_url = getattr(settings, "redis_url", "redis://localhost:6379/0")
        _state_manager = WorkflowStateManager(redis_url)

    if not _engine:
        _engine = WorkflowEngine(_state_manager)

    return _engine


@router.post(
    "/create",
    response_model=WorkflowDefinition,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workflow definition",
)
async def create_workflow(request: WorkflowCreateRequest) -> WorkflowDefinition:
    """
    Create a new workflow definition from YAML or JSON.

    The workflow definition should include:
    - Workflow metadata (name, description, version)
    - Step definitions with actions and conditions
    - Initial step to start execution
    - Retry policies and timeout configurations

    Example workflow definition:
    ```yaml
    id: data-processing-workflow
    name: Data Processing Pipeline
    version: 1.0.0
    initial_step: fetch_data
    steps:
      - id: fetch_data
        name: Fetch Data from API
        type: task
        action: http_request
        params:
          url: https://api.example.com/data
        timeout_seconds: 30
        next_on_success: [process_data]
        next_on_failure: [error_handler]
      - id: process_data
        name: Process Data
        type: task
        action: transform_data
        next_on_success: [save_results]
      - id: save_results
        name: Save Results
        type: task
        action: save_to_database
    ```
    """
    try:
        engine = get_workflow_engine()

        # Parse definition (support both YAML and JSON)
        if isinstance(request.definition, dict):
            definition_dict = request.definition
        else:
            definition_dict = yaml.safe_load(request.definition)

        # Ensure ID is set
        if "id" not in definition_dict:
            definition_dict["id"] = str(uuid4())

        # Create workflow definition
        definition = WorkflowDefinition(
            name=request.name,
            description=request.description,
            version=request.version,
            **definition_dict,
        )

        # Register workflow
        await engine.register_workflow(definition)

        logger.info(f"Created workflow: {definition.name} (ID: {definition.id})")
        return definition

    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workflow definition: {str(e)}",
        )


@router.post(
    "/{workflow_id}/execute",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Execute a workflow",
)
async def execute_workflow(
    workflow_id: str, request: WorkflowExecuteRequest
) -> dict:
    """
    Start execution of a workflow with optional input variables.

    The workflow will execute asynchronously. Use the returned execution_id
    to check status via the /workflows/{id}/status endpoint.

    Supports:
    - Input variables for workflow execution
    - Webhook notifications for state changes
    - Conditional step execution
    - Parallel step execution
    - Automatic retry on failure

    Returns:
        execution_id: Unique identifier for this workflow execution
        state: Initial execution state (PENDING)
    """
    try:
        engine = get_workflow_engine()

        execution = await engine.start_execution(
            workflow_id=workflow_id,
            variables=request.variables,
            webhook_url=request.webhook_url,
            webhook_events=request.webhook_events,
        )

        logger.info(f"Started workflow execution: {execution.id}")

        return {
            "execution_id": execution.id,
            "workflow_id": workflow_id,
            "state": execution.state.value,
            "message": "Workflow execution started",
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow execution: {str(e)}",
        )


@router.get(
    "/{execution_id}/status",
    response_model=WorkflowStatusResponse,
    summary="Get workflow execution status",
)
async def get_workflow_status(execution_id: str) -> WorkflowStatusResponse:
    """
    Get the current status of a workflow execution.

    Returns detailed information including:
    - Current execution state (PENDING, RUNNING, COMPLETED, FAILED)
    - Current step being executed
    - Progress percentage based on completed steps
    - Execution timing information
    - Error details if failed

    States:
    - PENDING: Workflow is queued for execution
    - RUNNING: Workflow is currently executing
    - PAUSED: Workflow is paused (can be resumed)
    - COMPLETED: Workflow finished successfully
    - FAILED: Workflow execution failed
    - CANCELLED: Workflow was cancelled by user
    """
    try:
        engine = get_workflow_engine()
        execution = await engine.get_execution_status(execution_id)

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution not found: {execution_id}",
            )

        # Calculate progress
        workflow = engine._workflows.get(execution.workflow_id)
        total_steps = len(workflow.steps) if workflow else 0
        completed_steps = len(
            [
                h
                for h in execution.history
                if "to_state" in h and h["to_state"] == "COMPLETED"
            ]
        )
        progress = (
            (completed_steps / total_steps * 100) if total_steps > 0 else 0.0
        )

        # Calculate elapsed time
        elapsed = None
        if execution.started_at:
            end_time = execution.completed_at or datetime.utcnow()
            elapsed = (end_time - execution.started_at).total_seconds()

        return WorkflowStatusResponse(
            execution_id=execution.id,
            workflow_id=execution.workflow_id,
            state=execution.state,
            current_step=execution.current_step,
            progress_percent=round(progress, 2),
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            elapsed_seconds=elapsed,
            steps_completed=completed_steps,
            steps_total=total_steps,
            error=execution.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve workflow status: {str(e)}",
        )


@router.post(
    "/{execution_id}/cancel",
    summary="Cancel a running workflow execution",
)
async def cancel_workflow(execution_id: str) -> dict:
    """
    Cancel a running workflow execution.

    Only workflows in PENDING or RUNNING state can be cancelled.
    This will stop execution and mark the workflow as CANCELLED.
    """
    try:
        engine = get_workflow_engine()
        success = await engine.cancel_execution(execution_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workflow cannot be cancelled (not found or already completed)",
            )

        return {
            "execution_id": execution_id,
            "state": WorkflowState.CANCELLED.value,
            "message": "Workflow execution cancelled",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel workflow: {str(e)}",
        )


@router.get(
    "/",
    summary="List workflow executions",
)
async def list_executions(
    workflow_id: Optional[str] = None,
    state: Optional[WorkflowState] = None,
    limit: int = 50,
) -> dict:
    """
    List workflow executions with optional filtering.

    Parameters:
    - workflow_id: Filter by specific workflow definition
    - state: Filter by execution state
    - limit: Maximum number of results (default: 50)

    Returns list of executions with basic information.
    """
    try:
        engine = get_workflow_engine()
        executions = await engine.state_manager.list_executions(workflow_id, state)

        # Limit results
        executions = executions[:limit]

        return {
            "total": len(executions),
            "executions": [
                {
                    "execution_id": e.id,
                    "workflow_id": e.workflow_id,
                    "state": e.state.value,
                    "created_at": e.created_at.isoformat(),
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "completed_at": (
                        e.completed_at.isoformat() if e.completed_at else None
                    ),
                }
                for e in executions
            ],
        }

    except Exception as e:
        logger.error(f"Failed to list executions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list executions: {str(e)}",
        )


@router.get(
    "/{execution_id}/history",
    summary="Get workflow execution history",
)
async def get_execution_history(execution_id: str) -> dict:
    """
    Get complete audit trail of workflow execution.

    Returns all state transitions with timestamps and metadata,
    providing full visibility into workflow execution flow.
    """
    try:
        engine = get_workflow_engine()
        history = await engine.state_manager.get_history(execution_id)

        return {
            "execution_id": execution_id,
            "total_transitions": len(history),
            "history": [
                {
                    "from_state": h.from_state.value
                    if hasattr(h.from_state, "value")
                    else h.from_state,
                    "to_state": h.to_state.value
                    if hasattr(h.to_state, "value")
                    else h.to_state,
                    "timestamp": h.timestamp.isoformat(),
                    "trigger": h.trigger,
                    "metadata": h.metadata,
                }
                for h in history
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get execution history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve execution history: {str(e)}",
        )


# Import datetime at module level
from datetime import datetime