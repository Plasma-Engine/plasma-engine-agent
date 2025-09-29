"""Comprehensive tests for workflow engine with state machine functionality."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.workflow import (
    RetryPolicy,
    StepCondition,
    StepState,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowState,
    WorkflowStep,
)
from app.workflows.engine import WorkflowEngine
from app.workflows.state_manager import WorkflowStateManager


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager for testing."""
    manager = AsyncMock(spec=WorkflowStateManager)
    manager.connect = AsyncMock()
    manager.disconnect = AsyncMock()
    manager.save_execution = AsyncMock()
    manager.get_execution = AsyncMock()
    manager.update_execution_state = AsyncMock()
    manager._save_transition = AsyncMock()
    return manager


@pytest.fixture
def simple_workflow():
    """Create a simple workflow definition for testing."""
    return WorkflowDefinition(
        id="test-workflow-1",
        name="Simple Test Workflow",
        description="A simple workflow for testing",
        version="1.0.0",
        initial_step="step1",
        steps=[
            WorkflowStep(
                id="step1",
                name="First Step",
                type="task",
                action="test_action",
                params={"key": "value"},
                next_on_success=["step2"],
                next_on_failure=["error_handler"],
            ),
            WorkflowStep(
                id="step2",
                name="Second Step",
                type="task",
                action="test_action_2",
                next_on_success=[],
            ),
            WorkflowStep(
                id="error_handler",
                name="Error Handler",
                type="task",
                action="handle_error",
                next_on_success=[],
            ),
        ],
    )


@pytest.fixture
def conditional_workflow():
    """Create a workflow with conditional steps."""
    return WorkflowDefinition(
        id="test-workflow-conditional",
        name="Conditional Workflow",
        version="1.0.0",
        initial_step="check",
        steps=[
            WorkflowStep(
                id="check",
                name="Check Condition",
                type="task",
                action="check",
                next_on_success=["conditional_step"],
            ),
            WorkflowStep(
                id="conditional_step",
                name="Conditional Step",
                type="task",
                action="conditional_action",
                condition=StepCondition(
                    expression="value > 10",
                    variables={"value": 15},
                ),
                next_on_success=["end"],
            ),
            WorkflowStep(
                id="end",
                name="End Step",
                type="task",
                action="end",
                next_on_success=[],
            ),
        ],
    )


@pytest.fixture
def parallel_workflow():
    """Create a workflow with parallel steps."""
    return WorkflowDefinition(
        id="test-workflow-parallel",
        name="Parallel Workflow",
        version="1.0.0",
        initial_step="start",
        steps=[
            WorkflowStep(
                id="start",
                name="Start Parallel",
                type="parallel",
                parallel_steps=["parallel1", "parallel2", "parallel3"],
                next_on_success=["end"],
            ),
            WorkflowStep(
                id="parallel1",
                name="Parallel Task 1",
                type="task",
                action="task1",
            ),
            WorkflowStep(
                id="parallel2",
                name="Parallel Task 2",
                type="task",
                action="task2",
            ),
            WorkflowStep(
                id="parallel3",
                name="Parallel Task 3",
                type="task",
                action="task3",
            ),
            WorkflowStep(
                id="end",
                name="End Step",
                type="task",
                action="end",
                next_on_success=[],
            ),
        ],
    )


@pytest.fixture
def retry_workflow():
    """Create a workflow with retry policy."""
    return WorkflowDefinition(
        id="test-workflow-retry",
        name="Retry Workflow",
        version="1.0.0",
        initial_step="retry_step",
        steps=[
            WorkflowStep(
                id="retry_step",
                name="Step with Retry",
                type="task",
                action="failing_action",
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    initial_delay_seconds=0.1,
                    backoff_multiplier=2.0,
                    max_delay_seconds=1.0,
                ),
                next_on_success=["success"],
                next_on_failure=["failure"],
            ),
            WorkflowStep(
                id="success",
                name="Success Step",
                type="task",
                action="success",
                next_on_success=[],
            ),
            WorkflowStep(
                id="failure",
                name="Failure Step",
                type="task",
                action="failure",
                next_on_success=[],
            ),
        ],
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestWorkflowEngine:
    """Test suite for WorkflowEngine."""

    async def test_register_workflow(self, mock_state_manager, simple_workflow):
        """Test workflow registration."""
        engine = WorkflowEngine(mock_state_manager)

        await engine.register_workflow(simple_workflow)

        assert simple_workflow.id in engine._workflows
        assert engine._workflows[simple_workflow.id] == simple_workflow

    async def test_start_execution(self, mock_state_manager, simple_workflow):
        """Test starting workflow execution."""
        engine = WorkflowEngine(mock_state_manager)
        await engine.register_workflow(simple_workflow)

        variables = {"input_data": "test"}
        execution = await engine.start_execution(
            workflow_id=simple_workflow.id,
            variables=variables,
        )

        assert execution.id is not None
        assert execution.workflow_id == simple_workflow.id
        assert execution.state == WorkflowState.PENDING
        assert execution.variables["input_data"] == "test"
        mock_state_manager.save_execution.assert_called_once()

    async def test_start_execution_nonexistent_workflow(self, mock_state_manager):
        """Test starting execution for non-existent workflow fails."""
        engine = WorkflowEngine(mock_state_manager)

        with pytest.raises(ValueError, match="Workflow not found"):
            await engine.start_execution(workflow_id="nonexistent")

    async def test_evaluate_condition_true(self, mock_state_manager):
        """Test condition evaluation returns True."""
        engine = WorkflowEngine(mock_state_manager)

        result = engine._evaluate_condition(
            "value > 10", {"value": 15}
        )

        assert result is True

    async def test_evaluate_condition_false(self, mock_state_manager):
        """Test condition evaluation returns False."""
        engine = WorkflowEngine(mock_state_manager)

        result = engine._evaluate_condition(
            "value > 10", {"value": 5}
        )

        assert result is False

    async def test_evaluate_condition_complex(self, mock_state_manager):
        """Test complex condition evaluation."""
        engine = WorkflowEngine(mock_state_manager)

        result = engine._evaluate_condition(
            "status == 'active' and count >= 3",
            {"status": "active", "count": 5},
        )

        assert result is True

    async def test_evaluate_condition_safe_eval(self, mock_state_manager):
        """Test that dangerous expressions are safely handled."""
        engine = WorkflowEngine(mock_state_manager)

        # Should return False on error, not execute dangerous code
        result = engine._evaluate_condition(
            "__import__('os').system('ls')", {}
        )

        assert result is False

    async def test_calculate_retry_delay(self, mock_state_manager):
        """Test retry delay calculation with exponential backoff."""
        engine = WorkflowEngine(mock_state_manager)
        retry_policy = RetryPolicy(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=10.0,
        )

        delay1 = engine._calculate_retry_delay(retry_policy, 1)
        delay2 = engine._calculate_retry_delay(retry_policy, 2)
        delay3 = engine._calculate_retry_delay(retry_policy, 3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    async def test_calculate_retry_delay_max_cap(self, mock_state_manager):
        """Test retry delay is capped at max_delay_seconds."""
        engine = WorkflowEngine(mock_state_manager)
        retry_policy = RetryPolicy(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=5.0,
        )

        delay = engine._calculate_retry_delay(retry_policy, 10)

        assert delay == 5.0

    async def test_get_next_step_success(self, mock_state_manager):
        """Test getting next step on success."""
        engine = WorkflowEngine(mock_state_manager)
        step = WorkflowStep(
            id="test",
            name="Test",
            type="task",
            next_on_success=["next1"],
            next_on_failure=["error"],
        )

        next_step = engine._get_next_step(step, success=True)

        assert next_step == "next1"

    async def test_get_next_step_failure(self, mock_state_manager):
        """Test getting next step on failure."""
        engine = WorkflowEngine(mock_state_manager)
        step = WorkflowStep(
            id="test",
            name="Test",
            type="task",
            next_on_success=["next1"],
            next_on_failure=["error"],
        )

        next_step = engine._get_next_step(step, success=False)

        assert next_step == "error"

    async def test_get_next_step_none(self, mock_state_manager):
        """Test getting next step when no next steps defined."""
        engine = WorkflowEngine(mock_state_manager)
        step = WorkflowStep(
            id="test",
            name="Test",
            type="task",
        )

        next_step = engine._get_next_step(step, success=True)

        assert next_step is None

    async def test_find_step(self, mock_state_manager, simple_workflow):
        """Test finding step by ID."""
        engine = WorkflowEngine(mock_state_manager)

        step = engine._find_step(simple_workflow, "step1")

        assert step is not None
        assert step.id == "step1"
        assert step.name == "First Step"

    async def test_find_step_not_found(self, mock_state_manager, simple_workflow):
        """Test finding non-existent step returns None."""
        engine = WorkflowEngine(mock_state_manager)

        step = engine._find_step(simple_workflow, "nonexistent")

        assert step is None

    async def test_execute_wait_step(self, mock_state_manager):
        """Test executing wait/delay step."""
        engine = WorkflowEngine(mock_state_manager)
        step = WorkflowStep(
            id="wait",
            name="Wait Step",
            type="wait",
            params={"duration_seconds": 0.1},
        )

        start = datetime.utcnow()
        await engine._execute_wait_step(step)
        end = datetime.utcnow()

        elapsed = (end - start).total_seconds()
        assert elapsed >= 0.1

    async def test_cancel_execution(self, mock_state_manager, simple_workflow):
        """Test cancelling a running workflow execution."""
        engine = WorkflowEngine(mock_state_manager)
        await engine.register_workflow(simple_workflow)

        # Create mock execution
        execution = WorkflowExecution(
            id="test-exec-1",
            workflow_id=simple_workflow.id,
            state=WorkflowState.RUNNING,
        )
        mock_state_manager.get_execution.return_value = execution

        # Create mock task
        mock_task = AsyncMock()
        engine._running_executions[execution.id] = mock_task

        success = await engine.cancel_execution(execution.id)

        assert success is True
        mock_task.cancel.assert_called_once()
        mock_state_manager.update_execution_state.assert_called_with(
            execution.id, WorkflowState.CANCELLED
        )

    async def test_cancel_execution_not_found(self, mock_state_manager):
        """Test cancelling non-existent execution returns False."""
        engine = WorkflowEngine(mock_state_manager)
        mock_state_manager.get_execution.return_value = None

        success = await engine.cancel_execution("nonexistent")

        assert success is False

    async def test_cancel_execution_already_completed(self, mock_state_manager):
        """Test cancelling completed execution returns False."""
        engine = WorkflowEngine(mock_state_manager)
        execution = WorkflowExecution(
            id="test-exec-1",
            workflow_id="test",
            state=WorkflowState.COMPLETED,
        )
        mock_state_manager.get_execution.return_value = execution

        success = await engine.cancel_execution(execution.id)

        assert success is False

    @patch("httpx.AsyncClient.post")
    async def test_send_webhook(self, mock_post, mock_state_manager):
        """Test sending webhook notifications."""
        engine = WorkflowEngine(mock_state_manager)

        execution = WorkflowExecution(
            id="test-exec-1",
            workflow_id="test",
            state=WorkflowState.COMPLETED,
            webhook_url="https://example.com/webhook",
            webhook_events=["completed"],
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        await engine._send_webhook(execution, "completed")

        mock_post.assert_called_once()

    @patch("httpx.AsyncClient.post")
    async def test_send_webhook_not_in_events(self, mock_post, mock_state_manager):
        """Test webhook not sent for events not in webhook_events."""
        engine = WorkflowEngine(mock_state_manager)

        execution = WorkflowExecution(
            id="test-exec-1",
            workflow_id="test",
            state=WorkflowState.RUNNING,
            webhook_url="https://example.com/webhook",
            webhook_events=["completed"],
        )

        await engine._send_webhook(execution, "started")

        mock_post.assert_not_called()

    async def test_get_execution_status(self, mock_state_manager):
        """Test retrieving execution status."""
        engine = WorkflowEngine(mock_state_manager)

        execution = WorkflowExecution(
            id="test-exec-1",
            workflow_id="test",
            state=WorkflowState.RUNNING,
        )
        mock_state_manager.get_execution.return_value = execution

        result = await engine.get_execution_status("test-exec-1")

        assert result == execution
        mock_state_manager.get_execution.assert_called_with("test-exec-1")


@pytest.mark.unit
class TestWorkflowValidation:
    """Test workflow definition validation."""

    def test_valid_workflow_definition(self):
        """Test creating valid workflow definition."""
        workflow = WorkflowDefinition(
            id="test",
            name="Test Workflow",
            version="1.0.0",
            initial_step="step1",
            steps=[
                WorkflowStep(
                    id="step1",
                    name="Step 1",
                    type="task",
                )
            ],
        )

        assert workflow.id == "test"
        assert workflow.name == "Test Workflow"

    def test_invalid_initial_step(self):
        """Test validation fails for invalid initial_step."""
        with pytest.raises(ValueError, match="not found in workflow steps"):
            WorkflowDefinition(
                id="test",
                name="Test Workflow",
                version="1.0.0",
                initial_step="nonexistent",
                steps=[
                    WorkflowStep(
                        id="step1",
                        name="Step 1",
                        type="task",
                    )
                ],
            )

    def test_invalid_version_format(self):
        """Test validation fails for invalid version format."""
        with pytest.raises(ValueError):
            WorkflowDefinition(
                id="test",
                name="Test Workflow",
                version="invalid",  # Should be semantic version
                initial_step="step1",
                steps=[
                    WorkflowStep(
                        id="step1",
                        name="Step 1",
                        type="task",
                    )
                ],
            )

    def test_step_condition_validation(self):
        """Test step condition expression validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            StepCondition(expression="", variables={})

    def test_step_condition_dangerous_keywords(self):
        """Test step condition rejects dangerous expressions."""
        with pytest.raises(ValueError, match="dangerous keywords"):
            StepCondition(expression="import os", variables={})

    def test_retry_policy_constraints(self):
        """Test retry policy validation constraints."""
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)  # Must be >= 1

        with pytest.raises(ValueError):
            RetryPolicy(backoff_multiplier=0.5)  # Must be >= 1.0


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowExecution:
    """Integration tests for complete workflow execution."""

    @pytest.fixture
    async def redis_state_manager(self):
        """Create real state manager with fakeredis for testing."""
        try:
            import fakeredis.aioredis

            manager = WorkflowStateManager("redis://localhost:6379/15")
            # Use fakeredis for testing
            manager._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
            yield manager
            await manager.disconnect()
        except ImportError:
            pytest.skip("fakeredis not installed")

    async def test_full_workflow_execution(self, redis_state_manager):
        """Test complete workflow execution flow."""
        engine = WorkflowEngine(redis_state_manager)

        # Create simple workflow
        workflow = WorkflowDefinition(
            id="integration-test-1",
            name="Integration Test",
            version="1.0.0",
            initial_step="start",
            steps=[
                WorkflowStep(
                    id="start",
                    name="Start",
                    type="task",
                    action="start_action",
                    next_on_success=["end"],
                ),
                WorkflowStep(
                    id="end",
                    name="End",
                    type="task",
                    action="end_action",
                    next_on_success=[],
                ),
            ],
        )

        await engine.register_workflow(workflow)
        execution = await engine.start_execution(workflow.id)

        # Wait for execution to complete
        await asyncio.sleep(0.5)

        # Check final state
        final_execution = await engine.get_execution_status(execution.id)
        assert final_execution is not None