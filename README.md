# Plasma Engine Agent

## Overview

**Plasma Engine Agent** is the autonomous AI agent orchestration and workflow automation service. It provides:

- 🤖 **Multi-Agent System**: Coordinated AI agents with specialized roles
- 🔄 **Workflow Engine**: Complex task orchestration and chaining
- 🧩 **Tool Integration**: Browser automation, API calls, file operations
- 🎯 **Goal Planning**: Task decomposition and execution strategies
- 💬 **MCP Support**: Model Context Protocol for tool discovery
- 📝 **Memory System**: Long-term context and learning

## Tech Stack

- **Language**: Python 3.11
- **Framework**: FastAPI + LangChain Agents
- **Orchestration**: Temporal / Prefect
- **Browser**: Playwright for web automation
- **MCP**: Model Context Protocol server/client
- **Queue**: Celery + Redis
- **Database**: PostgreSQL + pgvector
- **Monitoring**: OpenTelemetry

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Install Playwright browsers
playwright install

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload

# Run tests
pytest

# Start Celery worker
celery -A app.tasks worker --loglevel=info
```

## Architecture

```
User Request → Task Planner → Agent Selection → Tool Execution → Result
                    ↓              ↓                ↓             ↓
               Decomposition   Capability      MCP/Browser    Validation
                              Matching         Automation
```

## Key Features

- **Autonomous Execution**: Self-directed task completion
- **Tool Discovery**: Dynamic capability expansion via MCP
- **Error Recovery**: Automatic retry and fallback strategies
- **Human-in-the-Loop**: Approval gates and oversight
- **Parallel Execution**: Concurrent agent operations
- **State Management**: Checkpoint and resume capabilities

## Agent Types

- **Research Agent**: Information gathering and synthesis
- **Code Agent**: Development and debugging tasks
- **Browser Agent**: Web navigation and data extraction
- **API Agent**: External service integration
- **Analysis Agent**: Data processing and insights
- **Creative Agent**: Content generation and design

## Tool Capabilities

- **Browser Automation**: Form filling, clicking, navigation
- **File Operations**: Read, write, transform documents
- **API Interactions**: REST, GraphQL, webhooks
- **Database Queries**: SQL execution and data manipulation
- **Shell Commands**: System operations and scripts
- **MCP Tools**: Dynamic tool loading and execution

## Workflow Examples

```python
# Example: Automated research workflow
workflow = {
    "name": "market_research",
    "steps": [
        {"agent": "browser", "action": "search", "params": {"query": "..."}},
        {"agent": "research", "action": "analyze", "params": {"sources": "..."}},
        {"agent": "creative", "action": "generate_report", "params": {"data": "..."}}
    ]
}
```

## Development

See [Development Handbook](../plasma-engine-shared/docs/development-handbook.md) for guidelines.

## CI/CD

This repository uses GitHub Actions for CI/CD. All PRs are automatically:
- Linted and tested
- Security scanned
- Reviewed by CodeRabbit

See `.github/workflows/ci.yml` for details.