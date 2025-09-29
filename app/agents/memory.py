"""Memory management for LangChain agents with persistence support."""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
)
from langchain.schema import BaseMemory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class MemoryType(str, Enum):
    """Types of memory supported by the agent system."""

    BUFFER = "buffer"  # Store all messages
    WINDOW = "window"  # Store last N messages
    SUMMARY = "summary"  # Summarize conversation
    SUMMARY_BUFFER = "summary_buffer"  # Hybrid approach


class MemoryConfig(BaseModel):
    """Configuration for agent memory."""

    memory_type: MemoryType = MemoryType.BUFFER
    max_token_limit: int = 2000
    window_size: int = 5
    return_messages: bool = True
    input_key: str = "input"
    output_key: str = "output"
    memory_key: str = "chat_history"


class MemoryManager:
    """Manager for agent memory with persistence and session handling."""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        storage_path: Optional[Path] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize memory manager.

        Args:
            config: Memory configuration
            storage_path: Path for persisting memory
            llm_model: LLM model for summary memory
        """
        self.config = config or MemoryConfig()
        self.storage_path = storage_path or Path("/tmp/agent_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.llm_model = llm_model
        self._sessions: Dict[str, BaseMemory] = {}

    def create_memory(
        self, session_id: str, openai_api_key: Optional[str] = None
    ) -> BaseMemory:
        """Create a new memory instance for a session.

        Args:
            session_id: Unique identifier for the session
            openai_api_key: OpenAI API key for summary memory

        Returns:
            Configured memory instance
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        if self.config.memory_type == MemoryType.BUFFER:
            memory = ConversationBufferMemory(
                return_messages=self.config.return_messages,
                input_key=self.config.input_key,
                output_key=self.config.output_key,
                memory_key=self.config.memory_key,
            )

        elif self.config.memory_type == MemoryType.WINDOW:
            memory = ConversationBufferWindowMemory(
                k=self.config.window_size,
                return_messages=self.config.return_messages,
                input_key=self.config.input_key,
                output_key=self.config.output_key,
                memory_key=self.config.memory_key,
            )

        elif self.config.memory_type == MemoryType.SUMMARY:
            if not openai_api_key:
                raise ValueError("OpenAI API key required for summary memory")

            llm = ChatOpenAI(
                model=self.llm_model,
                temperature=0,
                openai_api_key=openai_api_key,
            )

            memory = ConversationSummaryMemory(
                llm=llm,
                return_messages=self.config.return_messages,
                input_key=self.config.input_key,
                output_key=self.config.output_key,
                memory_key=self.config.memory_key,
            )

        elif self.config.memory_type == MemoryType.SUMMARY_BUFFER:
            if not openai_api_key:
                raise ValueError("OpenAI API key required for summary buffer memory")

            llm = ChatOpenAI(
                model=self.llm_model,
                temperature=0,
                openai_api_key=openai_api_key,
            )

            memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=self.config.max_token_limit,
                return_messages=self.config.return_messages,
                input_key=self.config.input_key,
                output_key=self.config.output_key,
                memory_key=self.config.memory_key,
            )

        else:
            raise ValueError(f"Unsupported memory type: {self.config.memory_type}")

        self._sessions[session_id] = memory
        return memory

    def get_memory(self, session_id: str) -> Optional[BaseMemory]:
        """Retrieve memory for a session.

        Args:
            session_id: Session identifier

        Returns:
            Memory instance or None if not found
        """
        return self._sessions.get(session_id)

    def clear_memory(self, session_id: str) -> bool:
        """Clear memory for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if memory was cleared, False otherwise
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its memory.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False otherwise
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            # Delete persisted memory if exists
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            return True
        return False

    def save_memory(self, session_id: str) -> bool:
        """Persist memory to disk.

        Args:
            session_id: Session identifier

        Returns:
            True if memory was saved, False otherwise
        """
        memory = self._sessions.get(session_id)
        if not memory:
            return False

        try:
            session_file = self.storage_path / f"{session_id}.json"
            memory_data = {
                "session_id": session_id,
                "memory_type": self.config.memory_type,
                "chat_history": memory.load_memory_variables({}),
            }

            with open(session_file, "w") as f:
                json.dump(memory_data, f, indent=2, default=str)

            return True
        except Exception:
            return False

    def load_memory(
        self, session_id: str, openai_api_key: Optional[str] = None
    ) -> Optional[BaseMemory]:
        """Load persisted memory from disk.

        Args:
            session_id: Session identifier
            openai_api_key: OpenAI API key for summary memory

        Returns:
            Loaded memory instance or None if not found
        """
        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r") as f:
                memory_data = json.load(f)

            # Create new memory instance
            memory = self.create_memory(session_id, openai_api_key)

            # Restore chat history
            chat_history = memory_data.get("chat_history", {})
            if chat_history:
                # For buffer memory, we can restore messages
                if hasattr(memory, "chat_memory"):
                    for msg in chat_history.get(self.config.memory_key, []):
                        if isinstance(msg, dict):
                            memory.chat_memory.add_message(msg)

            return memory
        except Exception:
            return None

    def list_sessions(self) -> List[str]:
        """List all active sessions.

        Returns:
            List of session identifiers
        """
        return list(self._sessions.keys())

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session.

        Args:
            session_id: Session identifier

        Returns:
            Session information or None if not found
        """
        memory = self._sessions.get(session_id)
        if not memory:
            return None

        memory_vars = memory.load_memory_variables({})

        return {
            "session_id": session_id,
            "memory_type": self.config.memory_type,
            "message_count": len(memory_vars.get(self.config.memory_key, [])),
            "has_persisted_data": (self.storage_path / f"{session_id}.json").exists(),
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        total_messages = 0
        for memory in self._sessions.values():
            memory_vars = memory.load_memory_variables({})
            total_messages += len(memory_vars.get(self.config.memory_key, []))

        return {
            "active_sessions": len(self._sessions),
            "total_messages": total_messages,
            "memory_type": self.config.memory_type,
            "storage_path": str(self.storage_path),
        }