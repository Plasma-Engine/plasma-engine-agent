"""Agent memory management system for storing conversation context and tool results."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import aiosqlite
from loguru import logger


class AgentMemory:
    """Persistent memory system for agent conversations and tool results."""

    def __init__(self, db_path: Union[str, Path] = "agent_memory.db"):
        """Initialize memory with SQLite database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, memory_type, key)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_session
                ON memories(session_id, memory_type)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id, created_at)
            """)

            await db.commit()
            logger.info(f"Memory database initialized at {self.db_path}")

    async def store(
        self,
        session_id: str,
        key: str,
        value: Any,
        memory_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a memory entry.

        Args:
            session_id: Session identifier
            key: Memory key
            value: Memory value (will be JSON serialized)
            memory_type: Type of memory (general, tool_result, context, etc.)
            metadata: Optional metadata dictionary
        """
        value_json = json.dumps(value, default=str)
        metadata_json = json.dumps(metadata) if metadata else None

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO memories
                (session_id, memory_type, key, value, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (session_id, memory_type, key, value_json, metadata_json))
            await db.commit()

        logger.debug(f"Stored memory: {session_id}/{memory_type}/{key}")

    async def retrieve(
        self,
        session_id: str,
        key: str,
        memory_type: str = "general"
    ) -> Optional[Any]:
        """Retrieve a specific memory entry.

        Args:
            session_id: Session identifier
            key: Memory key
            memory_type: Type of memory

        Returns:
            Stored value or None if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT value FROM memories
                WHERE session_id = ? AND memory_type = ? AND key = ?
            """, (session_id, memory_type, key))

            row = await cursor.fetchone()
            if row:
                return json.loads(row[0])

        return None

    async def retrieve_all(
        self,
        session_id: str,
        memory_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve all memories for a session.

        Args:
            session_id: Session identifier
            memory_type: Optional memory type filter

        Returns:
            Dictionary of key-value pairs
        """
        query = "SELECT key, value FROM memories WHERE session_id = ?"
        params = [session_id]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        query += " ORDER BY updated_at DESC"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return {key: json.loads(value) for key, value in rows}

    async def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Store a conversation message.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            tool_calls: Optional tool calls data
        """
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversations
                (session_id, role, content, tool_calls)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, tool_calls_json))
            await db.commit()

        logger.debug(f"Stored message: {session_id}/{role}")

    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages

        Returns:
            List of conversation messages
        """
        query = """
            SELECT role, content, tool_calls, created_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY created_at ASC
        """

        params = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        messages = []
        for role, content, tool_calls, created_at in rows:
            message = {
                "role": role,
                "content": content,
                "created_at": created_at
            }
            if tool_calls:
                message["tool_calls"] = json.loads(tool_calls)
            messages.append(message)

        return messages

    async def clear_session(self, session_id: str) -> None:
        """Clear all memories and conversations for a session.

        Args:
            session_id: Session identifier to clear
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            await db.commit()

        logger.info(f"Cleared session: {session_id}")

    async def get_memory_stats(self) -> Dict[str, int]:
        """Get memory usage statistics.

        Returns:
            Statistics dictionary
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get total memories count
            cursor = await db.execute("SELECT COUNT(*) FROM memories")
            memories_count = (await cursor.fetchone())[0]

            # Get total conversations count
            cursor = await db.execute("SELECT COUNT(*) FROM conversations")
            conversations_count = (await cursor.fetchone())[0]

            # Get unique sessions count
            cursor = await db.execute("SELECT COUNT(DISTINCT session_id) FROM memories")
            sessions_count = (await cursor.fetchone())[0]

        return {
            "memories_count": memories_count,
            "conversations_count": conversations_count,
            "sessions_count": sessions_count
        }

    @asynccontextmanager
    async def session_context(self, session_id: str):
        """Context manager for session-scoped memory operations.

        Args:
            session_id: Session identifier

        Yields:
            Memory instance configured for the session
        """
        class SessionMemory:
            def __init__(self, memory: AgentMemory, session_id: str):
                self.memory = memory
                self.session_id = session_id

            async def store(self, key: str, value: Any, memory_type: str = "general", metadata: Optional[Dict] = None):
                return await self.memory.store(self.session_id, key, value, memory_type, metadata)

            async def retrieve(self, key: str, memory_type: str = "general"):
                return await self.memory.retrieve(self.session_id, key, memory_type)

            async def retrieve_all(self, memory_type: Optional[str] = None):
                return await self.memory.retrieve_all(self.session_id, memory_type)

            async def store_message(self, role: str, content: str, tool_calls: Optional[List[Dict]] = None):
                return await self.memory.store_message(self.session_id, role, content, tool_calls)

            async def get_conversation_history(self, limit: Optional[int] = None):
                return await self.memory.get_conversation_history(self.session_id, limit)

        try:
            yield SessionMemory(self, session_id)
        finally:
            pass  # Cleanup if needed