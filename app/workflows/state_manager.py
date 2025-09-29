"""State manager for workflow persistence using Redis."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from loguru import logger

from ..models.workflow import (
    StateTransition,
    StepState,
    WorkflowExecution,
    WorkflowState,
    WorkflowStep,
)


class WorkflowStateManager:
    """Manages workflow state persistence in Redis with audit trail."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        """Initialize state manager with Redis connection."""
        self.redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        if not self._redis:
            self._redis = await aioredis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
            logger.info("Connected to Redis for workflow state management")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")

    def _execution_key(self, execution_id: str) -> str:
        """Generate Redis key for execution."""
        return f"workflow:execution:{execution_id}"

    def _history_key(self, execution_id: str) -> str:
        """Generate Redis key for execution history."""
        return f"workflow:history:{execution_id}"

    def _workflow_index_key(self, workflow_id: str) -> str:
        """Generate Redis key for workflow index."""
        return f"workflow:index:{workflow_id}"

    async def save_execution(self, execution: WorkflowExecution) -> None:
        """Save workflow execution state to Redis."""
        if not self._redis:
            await self.connect()

        key = self._execution_key(execution.id)
        data = execution.model_dump(mode="json")

        # Convert datetime objects to ISO format strings
        for field in ["created_at", "started_at", "completed_at"]:
            if data.get(field):
                if isinstance(data[field], datetime):
                    data[field] = data[field].isoformat()

        await self._redis.set(key, json.dumps(data))

        # Add to workflow index
        index_key = self._workflow_index_key(execution.workflow_id)
        await self._redis.sadd(index_key, execution.id)

        logger.debug(f"Saved execution state: {execution.id} - {execution.state}")

    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Retrieve workflow execution from Redis."""
        if not self._redis:
            await self.connect()

        key = self._execution_key(execution_id)
        data = await self._redis.get(key)

        if not data:
            return None

        execution_dict = json.loads(data)

        # Convert ISO strings back to datetime
        for field in ["created_at", "started_at", "completed_at"]:
            if execution_dict.get(field):
                execution_dict[field] = datetime.fromisoformat(execution_dict[field])

        return WorkflowExecution(**execution_dict)

    async def update_execution_state(
        self,
        execution_id: str,
        new_state: WorkflowState,
        current_step: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update execution state with transition tracking."""
        execution = await self.get_execution(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        # Record state transition
        transition = StateTransition(
            from_state=execution.state,
            to_state=new_state,
            trigger=f"State change: {execution.state} -> {new_state}",
            metadata={
                "current_step": current_step,
                "error": error,
            },
        )

        execution.state = new_state
        if current_step:
            execution.current_step = current_step
        if error:
            execution.error = error
        if result:
            execution.result = result

        # Update timestamps
        if new_state == WorkflowState.RUNNING and not execution.started_at:
            execution.started_at = datetime.utcnow()
        elif new_state in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED]:
            execution.completed_at = datetime.utcnow()

        # Add to history
        execution.history.append(transition.model_dump(mode="json"))

        await self.save_execution(execution)
        await self._save_transition(execution_id, transition)

    async def _save_transition(
        self, execution_id: str, transition: StateTransition
    ) -> None:
        """Save state transition to history."""
        if not self._redis:
            await self.connect()

        history_key = self._history_key(execution_id)
        transition_data = transition.model_dump(mode="json")

        # Convert datetime to ISO string
        if isinstance(transition_data["timestamp"], datetime):
            transition_data["timestamp"] = transition_data["timestamp"].isoformat()

        await self._redis.rpush(history_key, json.dumps(transition_data))
        # Keep history for 30 days
        await self._redis.expire(history_key, 30 * 24 * 60 * 60)

    async def get_history(self, execution_id: str) -> List[StateTransition]:
        """Retrieve complete execution history."""
        if not self._redis:
            await self.connect()

        history_key = self._history_key(execution_id)
        transitions = await self._redis.lrange(history_key, 0, -1)

        result = []
        for t in transitions:
            data = json.loads(t)
            if data.get("timestamp"):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            result.append(StateTransition(**data))

        return result

    async def list_executions(
        self, workflow_id: Optional[str] = None, state: Optional[WorkflowState] = None
    ) -> List[WorkflowExecution]:
        """List workflow executions with optional filtering."""
        if not self._redis:
            await self.connect()

        executions = []

        if workflow_id:
            # Get from workflow index
            index_key = self._workflow_index_key(workflow_id)
            execution_ids = await self._redis.smembers(index_key)
        else:
            # Scan all execution keys
            cursor = 0
            execution_ids = []
            while True:
                cursor, keys = await self._redis.scan(
                    cursor, match="workflow:execution:*", count=100
                )
                execution_ids.extend([k.replace("workflow:execution:", "") for k in keys])
                if cursor == 0:
                    break

        for exec_id in execution_ids:
            execution = await self.get_execution(exec_id)
            if execution and (not state or execution.state == state):
                executions.append(execution)

        return executions

    async def delete_execution(self, execution_id: str) -> bool:
        """Delete workflow execution and its history."""
        if not self._redis:
            await self.connect()

        execution = await self.get_execution(execution_id)
        if not execution:
            return False

        # Remove from workflow index
        index_key = self._workflow_index_key(execution.workflow_id)
        await self._redis.srem(index_key, execution_id)

        # Delete execution and history
        exec_key = self._execution_key(execution_id)
        hist_key = self._history_key(execution_id)

        await self._redis.delete(exec_key, hist_key)
        logger.info(f"Deleted execution: {execution_id}")

        return True

    async def cleanup_old_executions(self, days: int = 30) -> int:
        """Clean up executions older than specified days."""
        if not self._redis:
            await self.connect()

        cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
        deleted = 0

        executions = await self.list_executions()
        for execution in executions:
            if execution.created_at.timestamp() < cutoff:
                if await self.delete_execution(execution.id):
                    deleted += 1

        logger.info(f"Cleaned up {deleted} old executions")
        return deleted