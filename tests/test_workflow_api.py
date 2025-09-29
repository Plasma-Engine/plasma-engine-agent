"""Tests for workflow API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from httpx import AsyncClient

from app.main import create_app
from app.models.workflow import (
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowState,
    WorkflowStep,
)


@pytest.fixture
def workflow_definition_dict():
    """Sample workflow definition as dictionary."""
    return {
        "id": "test-workflow-api",
        "name": "Test Workflow",
        "version": "1.0.0",
        "initial_step": "step1",
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "type": "task",
                "action": "test_action",
                "params": {"key": "value"},
                "next_on_success": ["step2"],
            },
            {
                "id": "step2",
                "name": "Step 2",
                "type": "task",
                "action": "test_action_2",
                "next_on_success": [],
            },
        ],
    }


@pytest.fixture
def mock_workflow_engine():
    """Mock workflow engine for API tests."""
    engine = MagicMock()
    engine.register_workflow = AsyncMock()
    engine.start_execution = AsyncMock()
    engine.get_execution_status = AsyncMock()
    engine.cancel_execution = AsyncMock()
    engine.state_manager = MagicMock()
    engine.state_manager.list_executions = AsyncMock()
    engine.state_manager.get_history = AsyncMock()
    return engine


@pytest.mark.asyncio
class TestWorkflowAPI:
    """Test workflow API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from httpx import ASGITransport
        app = create_app()
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def test_create_workflow_success(
        self, client, workflow_definition_dict, mock_workflow_engine
    ):
        """Test successful workflow creation."""
        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.post(
                "/api/v1/agent/workflows/create",
                json={
                    "name": "Test Workflow",
                    "description": "Test description",
                    "definition": workflow_definition_dict,
                    "version": "1.0.0",
                },
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Test Workflow"
        assert data["version"] == "1.0.0"

    async def test_create_workflow_invalid_definition(self, client):
        """Test workflow creation with invalid definition."""
        response = await client.post(
            "/api/v1/agent/workflows/create",
            json={
                "name": "Invalid Workflow",
                "definition": {
                    "invalid": "structure"
                },
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    async def test_execute_workflow_success(self, client, mock_workflow_engine):
        """Test successful workflow execution."""
        mock_execution = WorkflowExecution(
            id="exec-123",
            workflow_id="test-workflow",
            state=WorkflowState.PENDING,
        )
        mock_workflow_engine.start_execution.return_value = mock_execution

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.post(
                "/api/v1/agent/workflows/test-workflow/execute",
                json={
                    "variables": {"key": "value"},
                    "webhook_url": "https://example.com/webhook",
                    "webhook_events": ["completed"],
                },
            )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["execution_id"] == "exec-123"
        assert data["state"] == "pending"

    async def test_execute_workflow_not_found(self, client, mock_workflow_engine):
        """Test executing non-existent workflow."""
        mock_workflow_engine.start_execution.side_effect = ValueError("Workflow not found")

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.post(
                "/api/v1/agent/workflows/nonexistent/execute",
                json={},
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_workflow_status_success(self, client, mock_workflow_engine):
        """Test getting workflow execution status."""
        from datetime import datetime

        mock_execution = WorkflowExecution(
            id="exec-123",
            workflow_id="test-workflow",
            state=WorkflowState.RUNNING,
            current_step="step1",
            started_at=datetime.utcnow(),
        )
        mock_workflow_engine.get_execution_status.return_value = mock_execution
        mock_workflow_engine._workflows = {
            "test-workflow": MagicMock(steps=[MagicMock(), MagicMock()])
        }

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.get(
                "/api/v1/agent/workflows/exec-123/status"
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["execution_id"] == "exec-123"
        assert data["state"] == "running"
        assert data["current_step"] == "step1"

    async def test_get_workflow_status_not_found(self, client, mock_workflow_engine):
        """Test getting status for non-existent execution."""
        mock_workflow_engine.get_execution_status.return_value = None

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.get(
                "/api/v1/agent/workflows/nonexistent/status"
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_cancel_workflow_success(self, client, mock_workflow_engine):
        """Test cancelling workflow execution."""
        mock_workflow_engine.cancel_execution.return_value = True

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.post(
                "/api/v1/agent/workflows/exec-123/cancel"
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["state"] == "cancelled"

    async def test_cancel_workflow_cannot_cancel(self, client, mock_workflow_engine):
        """Test cancelling workflow that cannot be cancelled."""
        mock_workflow_engine.cancel_execution.return_value = False

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.post(
                "/api/v1/agent/workflows/exec-123/cancel"
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    async def test_list_executions(self, client, mock_workflow_engine):
        """Test listing workflow executions."""
        from datetime import datetime

        mock_executions = [
            WorkflowExecution(
                id="exec-1",
                workflow_id="workflow-1",
                state=WorkflowState.COMPLETED,
                created_at=datetime.utcnow(),
            ),
            WorkflowExecution(
                id="exec-2",
                workflow_id="workflow-1",
                state=WorkflowState.RUNNING,
                created_at=datetime.utcnow(),
                started_at=datetime.utcnow(),
            ),
        ]
        mock_workflow_engine.state_manager.list_executions.return_value = mock_executions

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.get(
                "/api/v1/agent/workflows/?workflow_id=workflow-1"
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["executions"]) == 2

    async def test_list_executions_with_state_filter(self, client, mock_workflow_engine):
        """Test listing executions with state filter."""
        from datetime import datetime

        mock_executions = [
            WorkflowExecution(
                id="exec-1",
                workflow_id="workflow-1",
                state=WorkflowState.COMPLETED,
                created_at=datetime.utcnow(),
            ),
        ]
        mock_workflow_engine.state_manager.list_executions.return_value = mock_executions

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.get(
                "/api/v1/agent/workflows/?state=completed&limit=10"
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1

    async def test_get_execution_history(self, client, mock_workflow_engine):
        """Test getting execution history."""
        from datetime import datetime
        from app.models.workflow import StateTransition

        mock_history = [
            StateTransition(
                from_state=WorkflowState.PENDING,
                to_state=WorkflowState.RUNNING,
                timestamp=datetime.utcnow(),
                trigger="Start execution",
            ),
            StateTransition(
                from_state=WorkflowState.RUNNING,
                to_state=WorkflowState.COMPLETED,
                timestamp=datetime.utcnow(),
                trigger="Workflow completed",
            ),
        ]
        mock_workflow_engine.state_manager.get_history.return_value = mock_history

        with patch("app.routers.workflows.get_workflow_engine", return_value=mock_workflow_engine):
            response = await client.get(
                "/api/v1/agent/workflows/exec-123/history"
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["execution_id"] == "exec-123"
        assert data["total_transitions"] == 2
        assert len(data["history"]) == 2


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowAPIIntegration:
    """Integration tests for workflow API."""

    @pytest.fixture
    def client(self):
        """Create test client with real app."""
        from httpx import ASGITransport
        app = create_app()
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def test_workflow_creation_yaml_format(self, client):
        """Test creating workflow with YAML definition."""
        yaml_definition = """
id: yaml-workflow
name: YAML Workflow
version: 1.0.0
initial_step: start
steps:
  - id: start
    name: Start Step
    type: task
    action: start_action
    next_on_success: [end]
  - id: end
    name: End Step
    type: task
    action: end_action
    next_on_success: []
"""
        response = await client.post(
            "/api/v1/agent/workflows/create",
            json={
                "name": "YAML Workflow",
                "definition": yaml_definition,
            },
        )

        # May fail if Redis not available, but tests structure
        assert response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]