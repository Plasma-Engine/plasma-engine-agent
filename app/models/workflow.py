"""Workflow models and schemas for state machine implementation."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class WorkflowState(str, Enum):
    """Workflow execution states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepState(str, Enum):
    """Individual step execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RetryPolicy(BaseModel):
    """Retry configuration for failed steps."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    initial_delay_seconds: float = Field(default=1.0, ge=0.1)
    max_delay_seconds: float = Field(default=300.0, ge=1.0)
    retry_on_states: List[str] = Field(default_factory=lambda: ["FAILED"])


class StepCondition(BaseModel):
    """Conditional execution logic for steps."""

    expression: str = Field(..., description="Python expression to evaluate")
    variables: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Ensure expression is safe and non-empty."""
        if not v or not v.strip():
            raise ValueError("Expression cannot be empty")
        # Basic safety check - prevent dangerous operations
        dangerous_keywords = ["import", "exec", "eval", "__", "open", "file"]
        if any(keyword in v.lower() for keyword in dangerous_keywords):
            raise ValueError(f"Expression contains dangerous keywords: {v}")
        return v


class WorkflowStep(BaseModel):
    """Individual workflow step definition."""

    id: str = Field(..., description="Unique step identifier")
    name: str = Field(..., description="Human-readable step name")
    type: str = Field(..., description="Step type (task, decision, parallel, wait)")
    action: Optional[str] = Field(None, description="Action to execute")
    params: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[StepCondition] = None
    retry_policy: Optional[RetryPolicy] = None
    timeout_seconds: Optional[float] = Field(None, ge=1.0)
    next_on_success: Optional[List[str]] = Field(
        None, description="Next step IDs on success"
    )
    next_on_failure: Optional[List[str]] = Field(
        None, description="Next step IDs on failure"
    )
    parallel_steps: Optional[List[str]] = Field(
        None, description="Steps to execute in parallel"
    )

    # Execution tracking
    state: StepState = Field(default=StepState.PENDING)
    attempts: int = Field(default=0, ge=0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class WorkflowDefinition(BaseModel):
    """Complete workflow definition with versioning."""

    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = None
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    steps: List[WorkflowStep] = Field(..., min_length=1)
    initial_step: str = Field(..., description="Starting step ID")
    variables: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("initial_step")
    @classmethod
    def validate_initial_step(cls, v: str, info: Any) -> str:
        """Ensure initial_step exists in steps."""
        if "steps" in info.data:
            step_ids = [step.id for step in info.data["steps"]]
            if v not in step_ids:
                raise ValueError(f"Initial step '{v}' not found in workflow steps")
        return v


class WorkflowExecution(BaseModel):
    """Runtime workflow execution state."""

    id: str = Field(..., description="Unique execution identifier")
    workflow_id: str = Field(..., description="Reference to workflow definition")
    workflow_version: str = Field(default="1.0.0")
    state: WorkflowState = Field(default=WorkflowState.PENDING)
    current_step: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Audit trail
    history: List[Dict[str, Any]] = Field(default_factory=list)

    # Webhook configuration
    webhook_url: Optional[str] = None
    webhook_events: List[str] = Field(default_factory=list)


class StateTransition(BaseModel):
    """State machine transition record."""

    from_state: Union[WorkflowState, StepState]
    to_state: Union[WorkflowState, StepState]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trigger: str = Field(..., description="What triggered the transition")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowCreateRequest(BaseModel):
    """API request for creating a workflow."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    definition: Dict[str, Any] = Field(..., description="YAML/JSON workflow definition")
    version: str = Field(default="1.0.0")


class WorkflowExecuteRequest(BaseModel):
    """API request for executing a workflow."""

    variables: Dict[str, Any] = Field(default_factory=dict)
    webhook_url: Optional[str] = None
    webhook_events: List[str] = Field(
        default_factory=lambda: ["completed", "failed"]
    )


class WorkflowStatusResponse(BaseModel):
    """API response for workflow status."""

    execution_id: str
    workflow_id: str
    state: WorkflowState
    current_step: Optional[str]
    progress_percent: float = Field(ge=0.0, le=100.0)
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    elapsed_seconds: Optional[float]
    steps_completed: int
    steps_total: int
    error: Optional[str] = None