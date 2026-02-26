"""Google Chat session management for ADK agents.

This module manages ADK session lifecycle for Google Chat users using
VertexAiSessionService. Each Google Chat user gets exactly one ADK session
that persists across all conversations (DMs, spaces, threads).

Pattern:
    - Session per user (not per space or thread)
    - Get-or-create on every message
    - Explicit reset via /reset command
    - Uses VertexAiSessionService for session CRUD
    - Uses /run_sse endpoint for agent invocation
    - Strategy pattern with dependency injection for testability

Architecture:
    Agent Foundation agent runs on Cloud Run (not Agent Engine runtime).
    Agent Engine provides session/memory persistence only.
    Session service configured with reasoning engine ID at initialization.
    Methods use ADK app name ("agent_foundation") not resource names.

Reference:
    Google's official pattern from travel-adk-ai-agent sample:
    https://github.com/googleworkspace/add-ons-samples/tree/main/python/travel-adk-ai-agent
"""

from __future__ import annotations

import logging

from google.adk.sessions import Session, VertexAiSessionService
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    ListSessionsResponse,
)
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def create_session_service(
    project: str,
    location: str,
    agent_engine_resource_name: str,
) -> BaseSessionService:
    """Factory for creating session service instances.

    Args:
        project: GCP project ID
        location: GCP region (e.g., "us-central1")
        agent_engine_resource_name: Full Agent Engine resource name
            (e.g., "projects/123/locations/us-central1/reasoningEngines/456")

    Returns:
        Configured session service instance (VertexAiSessionService)

    Example:
        >>> service = create_session_service(
        ...     "my-project",
        ...     "us-central1",
        ...     "projects/123/locations/us-central1/reasoningEngines/456"
        ... )
    """
    # Extract just the reasoning engine ID from the resource name
    # "projects/.../reasoningEngines/456" → "456"
    agent_engine_id = agent_engine_resource_name.split("/")[-1]

    return VertexAiSessionService(
        project=project, location=location, agent_engine_id=agent_engine_id
    )


class GoogleChatSessionManager:
    """Manages ADK sessions for Google Chat users.

    Implements session lifecycle management with dependency injection for
    testability. Encapsulates all session operations for Google Chat integration.

    Example:
        >>> service = create_session_service("project", "location", "resource_name")
        >>> manager = GoogleChatSessionManager(service, "agent_foundation")
        >>> session_id = await manager.get_or_create_session("users/123")
    """

    def __init__(self, session_service: BaseSessionService, app_name: str) -> None:
        """Initialize session manager with ADK session service.

        Args:
            session_service: Any ADK session service implementation
            app_name: ADK application name (e.g., "agent_foundation")
        """
        self._session_service = session_service
        self._app_name = app_name

    @staticmethod
    def extract_user_id(chat_user_name: str) -> str:
        """Extract user ID from Google Chat user resource name.

        Google Chat provides user identifiers in resource name format:
        "users/123456789". This function extracts just the numeric ID.

        Args:
            chat_user_name: Full Google Chat user resource name

        Returns:
            Clean user ID (e.g., "123456789")

        Examples:
            >>> GoogleChatSessionManager.extract_user_id("users/123456789")
            "123456789"
        """
        return chat_user_name.replace("users/", "")

    async def list_sessions(self, chat_user_name: str) -> list[str]:
        """List all session IDs for a Google Chat user.

        Args:
            chat_user_name: Google Chat user resource name (e.g., "users/123456789")

        Returns:
            List of session IDs (empty list if user has no sessions)

        Examples:
            >>> sessions = await manager.list_sessions("users/123")
            >>> # Returns: ["session-abc-123", "session-xyz-456"]
        """
        user_id = self.extract_user_id(chat_user_name)
        logger.debug(f"Listing sessions for user {user_id}")

        result: ListSessionsResponse = await self._session_service.list_sessions(
            app_name=self._app_name,
            user_id=user_id,
        )

        if not result or not result.sessions:
            logger.debug(f"No sessions found for user {user_id}")
            return []

        session_ids = [session.id for session in result.sessions]
        logger.debug(f"Found {len(session_ids)} session(s) for user {user_id}")
        return session_ids

    async def create_session(self, chat_user_name: str) -> str:
        """Create new ADK session for a Google Chat user.

        Args:
            chat_user_name: Google Chat user resource name

        Returns:
            New session ID

        Examples:
            >>> session_id = await manager.create_session("users/123")
            >>> # Returns: "session-new-123"
        """
        user_id = self.extract_user_id(chat_user_name)
        logger.info(f"Creating new session for user {user_id}")

        session: Session = await self._session_service.create_session(
            app_name=self._app_name,
            user_id=user_id,
        )

        logger.info(f"Created session {session.id} for user {user_id}")
        return session.id

    async def get_or_create_session(self, chat_user_name: str) -> str:
        """Get existing ADK session for user or create new one if none exists.

        Implements get-or-create pattern from Google's reference implementation:
        1. List existing sessions for user
        2. If found, return first session ID (users typically have one)
        3. If not found, create new session and return its ID

        This ensures conversation continuity while avoiding session duplication.

        Args:
            chat_user_name: Google Chat user resource name

        Returns:
            Session ID (existing or newly created)

        Examples:
            >>> # First call - creates session
            >>> session_id = await manager.get_or_create_session("users/123")
            >>> # Returns: "session-abc-123"
            >>>
            >>> # Second call - reuses session
            >>> session_id = await manager.get_or_create_session("users/123")
            >>> # Returns: "session-abc-123" (same ID)
        """
        with tracer.start_as_current_span(
            "session.get_or_create",
            attributes={"chat.user.name": chat_user_name},
        ) as span:
            user_id = self.extract_user_id(chat_user_name)
            span.set_attribute("chat.user.id", user_id)

            sessions = await self.list_sessions(chat_user_name)

            if sessions:
                session_id = sessions[0]
                span.set_attribute("session.created", False)
                span.set_attribute("session.id", session_id)
                logger.debug(f"Reusing session {session_id} for user {user_id}")
                return session_id

            session_id = await self.create_session(chat_user_name)
            span.set_attribute("session.created", True)
            span.set_attribute("session.id", session_id)
            return session_id

    async def delete_session(self, chat_user_name: str, session_id: str) -> None:
        """Delete specific ADK session for a user.

        Args:
            chat_user_name: Google Chat user resource name
            session_id: Session ID to delete

        Examples:
            >>> await manager.delete_session("users/123", "session-abc-123")
        """
        user_id = self.extract_user_id(chat_user_name)
        logger.info(f"Deleting session {session_id} for user {user_id}")

        await self._session_service.delete_session(
            app_name=self._app_name,
            user_id=user_id,
            session_id=session_id,
        )

        logger.info(f"Deleted session {session_id} for user {user_id}")

    async def delete_all_sessions(self, chat_user_name: str) -> int:
        """Delete all ADK sessions for a user (implements /reset command).

        Args:
            chat_user_name: Google Chat user resource name

        Returns:
            Number of sessions deleted

        Examples:
            >>> count = await manager.delete_all_sessions("users/123")
            >>> # Returns: 1 (deleted one session)
        """
        with tracer.start_as_current_span(
            "session.delete_all",
            attributes={"chat.user.name": chat_user_name},
        ) as span:
            user_id = self.extract_user_id(chat_user_name)
            span.set_attribute("chat.user.id", user_id)

            sessions = await self.list_sessions(chat_user_name)

            if not sessions:
                logger.debug(f"No sessions to delete for user {user_id}")
                span.set_attribute("sessions.deleted_count", 0)
                span.set_attribute("sessions.total_count", 0)
                return 0

            deleted_count = 0
            for session_id in sessions:
                with tracer.start_as_current_span("session.delete") as delete_span:
                    delete_span.set_attribute("session.id", session_id)
                    try:
                        await self.delete_session(chat_user_name, session_id)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete session {session_id}: {e}")
                        delete_span.set_status(trace.Status(trace.StatusCode.ERROR))
                        delete_span.record_exception(e)

            span.set_attribute("sessions.deleted_count", deleted_count)
            span.set_attribute("sessions.total_count", len(sessions))

            logger.info(
                f"Deleted {deleted_count}/{len(sessions)} sessions for user {user_id}"
            )
            return deleted_count
