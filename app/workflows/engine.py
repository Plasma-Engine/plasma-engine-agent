"""Workflow execution engine with state machine implementation."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models.workflow import (
    RetryPolicy,
    StepState,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowState,
    WorkflowStep,
)
from .state_manager import WorkflowStateManager


class WorkflowEngine:
    """
    State machine-based workflow execution engine.

    Features:
    - State machine with transitions (pending → running → completed/failed)
    - Conditional step execution with safe expression evaluation
    - Parallel step execution support
    - Error handling with configurable retry policies
    - Step timeout handling
    - Event-driven architecture with webhooks
    """

    def __init__(self, state_manager: WorkflowStateManager) -> None:
        """Initialize workflow engine with state manager."""
        self.state_manager = state_manager
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._running_executions: Dict[str, asyncio.Task] = {}

    async def register_workflow(self, definition: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._workflows[definition.id] = definition
        logger.info(
            f"Registered workflow: {definition.name} "
            f"(ID: {definition.id}, Version: {definition.version})"
        )

    async def start_execution(
        self,
        workflow_id: str,
        variables: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
        webhook_events: Optional[List[str]] = None,
    ) -> WorkflowExecution:
        """Start a new workflow execution."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Create execution
        execution = WorkflowExecution(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            workflow_version=workflow.version,
            state=WorkflowState.PENDING,
            current_step=workflow.initial_step,
            variables={**workflow.variables, **(variables or {})},
            webhook_url=webhook_url,
            webhook_events=webhook_events or [],
        )

        await self.state_manager.save_execution(execution)
        logger.info(f"Created workflow execution: {execution.id}")

        # Start execution in background
        task = asyncio.create_task(self._execute_workflow(execution.id))
        self._running_executions[execution.id] = task

        return execution

    async def _execute_workflow(self, execution_id: str) -> None:
        """Execute workflow with state machine transitions."""
        try:
            execution = await self.state_manager.get_execution(execution_id)
            if not execution:
                raise ValueError(f"Execution not found: {execution_id}")

            workflow = self._workflows.get(execution.workflow_id)
            if not workflow:
                raise ValueError(f"Workflow not found: {execution.workflow_id}")

            # Transition to RUNNING
            await self.state_manager.update_execution_state(
                execution_id, WorkflowState.RUNNING
            )
            await self._send_webhook(execution, "started")

            # Execute workflow steps
            step_id = execution.current_step
            while step_id:
                step = self._find_step(workflow, step_id)
                if not step:
                    raise ValueError(f"Step not found: {step_id}")

                # Execute step
                next_step = await self._execute_step(execution_id, workflow, step)
                step_id = next_step

            # Workflow completed successfully
            await self.state_manager.update_execution_state(
                execution_id, WorkflowState.COMPLETED
            )
            await self._send_webhook(execution, "completed")

            logger.info(f"Workflow execution completed: {execution_id}")

        except Exception as e:
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
            await self.state_manager.update_execution_state(
                execution_id, WorkflowState.FAILED, error=str(e)
            )
            execution = await self.state_manager.get_execution(execution_id)
            if execution:
                await self._send_webhook(execution, "failed")

        finally:
            if execution_id in self._running_executions:
                del self._running_executions[execution_id]

    async def _execute_step(
        self,
        execution_id: str,
        workflow: WorkflowDefinition,
        step: WorkflowStep,
    ) -> Optional[str]:
        """
        Execute a single workflow step with retry logic.

        Returns:
            Next step ID to execute, or None if workflow should end
        """
        execution = await self.state_manager.get_execution(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        logger.info(f"Executing step: {step.name} (ID: {step.id})")

        # Update current step
        await self.state_manager.update_execution_state(
            execution_id, WorkflowState.RUNNING, current_step=step.id
        )

        # Check condition
        if step.condition and not self._evaluate_condition(
            step.condition.expression, {**execution.variables, **step.condition.variables}
        ):
            logger.info(f"Step condition not met, skipping: {step.id}")
            return self._get_next_step(step, success=True)

        # Handle different step types
        try:
            if step.type == "parallel":
                await self._execute_parallel_steps(execution_id, workflow, step)
            elif step.type == "wait":
                await self._execute_wait_step(step)
            else:
                await self._execute_task_step(execution_id, step)

            return self._get_next_step(step, success=True)

        except Exception as e:
            logger.error(f"Step execution failed: {step.id} - {e}")

            # Handle retry policy
            if step.retry_policy and step.attempts < step.retry_policy.max_attempts:
                step.attempts += 1
                delay = self._calculate_retry_delay(step.retry_policy, step.attempts)
                logger.info(
                    f"Retrying step {step.id} in {delay}s "
                    f"(attempt {step.attempts}/{step.retry_policy.max_attempts})"
                )
                await asyncio.sleep(delay)
                return step.id  # Retry same step

            # Max retries exceeded
            step.state = StepState.FAILED
            step.error = str(e)
            return self._get_next_step(step, success=False)

    async def _execute_task_step(
        self, execution_id: str, step: WorkflowStep
    ) -> Dict[str, Any]:
        """Execute a task step with timeout handling."""
        execution = await self.state_manager.get_execution(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        step.state = StepState.RUNNING
        step.started_at = datetime.utcnow()

        try:
            # Execute step action with timeout
            timeout = step.timeout_seconds or 300.0  # Default 5 minutes

            result = await asyncio.wait_for(
                self._execute_action(step.action, step.params, execution.variables),
                timeout=timeout,
            )

            step.state = StepState.COMPLETED
            step.completed_at = datetime.utcnow()
            step.result = result

            # Update execution variables with step result
            if result:
                execution.variables.update(result)
                await self.state_manager.save_execution(execution)

            logger.info(f"Step completed: {step.id}")
            return result

        except asyncio.TimeoutError:
            step.state = StepState.FAILED
            step.error = f"Step timeout after {step.timeout_seconds}s"
            raise

        except Exception as e:
            step.state = StepState.FAILED
            step.error = str(e)
            raise

    async def _execute_parallel_steps(
        self,
        execution_id: str,
        workflow: WorkflowDefinition,
        step: WorkflowStep,
    ) -> None:
        """Execute multiple steps in parallel."""
        if not step.parallel_steps:
            return

        logger.info(f"Executing {len(step.parallel_steps)} parallel steps")

        # Create tasks for all parallel steps
        tasks = []
        for parallel_step_id in step.parallel_steps:
            parallel_step = self._find_step(workflow, parallel_step_id)
            if parallel_step:
                task = asyncio.create_task(
                    self._execute_step(execution_id, workflow, parallel_step)
                )
                tasks.append(task)

        # Wait for all parallel steps to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            raise RuntimeError(
                f"Parallel execution failed: {len(failures)} steps failed"
            )

    async def _execute_wait_step(self, step: WorkflowStep) -> None:
        """Execute a wait/delay step."""
        wait_seconds = step.params.get("duration_seconds", 1.0)
        logger.info(f"Waiting for {wait_seconds}s")
        await asyncio.sleep(wait_seconds)

    async def _execute_action(
        self,
        action: Optional[str],
        params: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a step action (placeholder for actual implementation)."""
        if not action:
            return {}

        # This is where you would integrate with actual task execution
        # For now, return the params as result
        logger.debug(f"Executing action: {action} with params: {params}")

        # Simulate work
        await asyncio.sleep(0.1)

        return {"action_executed": action, "params": params, "status": "success"}

    def _evaluate_condition(
        self, expression: str, variables: Dict[str, Any]
    ) -> bool:
        """
        Safely evaluate a conditional expression.

        Uses a restricted eval environment for safety.
        """
        try:
            # Create safe evaluation namespace
            safe_dict = {
                "__builtins__": {},
                "True": True,
                "False": False,
                "None": None,
                **variables,
            }

            result = eval(expression, safe_dict)
            return bool(result)

        except Exception as e:
            logger.error(f"Condition evaluation failed: {expression} - {e}")
            return False

    def _calculate_retry_delay(
        self, retry_policy: RetryPolicy, attempt: int
    ) -> float:
        """Calculate exponential backoff delay for retry."""
        delay = retry_policy.initial_delay_seconds * (
            retry_policy.backoff_multiplier ** (attempt - 1)
        )
        return min(delay, retry_policy.max_delay_seconds)

    def _get_next_step(self, step: WorkflowStep, success: bool) -> Optional[str]:
        """Determine next step based on execution result."""
        next_steps = step.next_on_success if success else step.next_on_failure

        if not next_steps:
            return None

        # For simplicity, return first next step
        # In a more complex implementation, this could handle branching logic
        return next_steps[0]

    def _find_step(
        self, workflow: WorkflowDefinition, step_id: str
    ) -> Optional[WorkflowStep]:
        """Find a step by ID in workflow definition."""
        for step in workflow.steps:
            if step.id == step_id:
                return step
        return None

    async def _send_webhook(self, execution: WorkflowExecution, event: str) -> None:
        """Send webhook notification for workflow events."""
        if not execution.webhook_url or event not in execution.webhook_events:
            return

        try:
            payload = {
                "event": event,
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
                "state": execution.state.value,
                "timestamp": datetime.utcnow().isoformat(),
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    execution.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info(f"Webhook sent: {event} to {execution.webhook_url}")

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get current execution status."""
        return await self.state_manager.get_execution(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        execution = await self.state_manager.get_execution(execution_id)
        if not execution:
            return False

        if execution.state not in [WorkflowState.PENDING, WorkflowState.RUNNING]:
            return False

        # Cancel running task
        if execution_id in self._running_executions:
            task = self._running_executions[execution_id]
            task.cancel()
            del self._running_executions[execution_id]

        # Update state
        await self.state_manager.update_execution_state(
            execution_id, WorkflowState.CANCELLED
        )

        logger.info(f"Cancelled workflow execution: {execution_id}")
        return True