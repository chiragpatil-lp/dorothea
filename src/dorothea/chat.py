"""Google Chat webhook integration.

This module provides webhook endpoints for Google Chat integration with the Agent
ADK agent. The webhook receives MESSAGE events from Google Chat, manages per-user
sessions, forwards messages to the ADK agent via /run_sse endpoint, and returns
formatted responses.

Architecture:
    Google Chat → /chat/webhook → get_or_create_session → ADK /run_sse → Response

Session Management:
    - One session per Google Chat user (not per space or thread)
    - Sessions persist across all user interactions
    - Users can reset via /reset command

Pattern:
    - APIRouter for route registration (FastAPI best practice)
    - Thin route wrapper for HTTP layer
    - Framework-agnostic business logic for easy testing
"""

import json
import logging
from contextlib import suppress
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from opentelemetry import trace

from .session import GoogleChatSessionManager, create_session_service

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Create router for Chat endpoints
chat_router = APIRouter(prefix="/chat", tags=["Google Chat Integration"])


def create_workspace_addon_response(text: str) -> dict[str, Any]:
    """Wrap text message in Google Workspace Add-on response format.

    Google Workspace Add-ons that extend Google Chat must return actions
    instead of plain Message objects. This function wraps a simple text
    message in the required CreateMessageAction format.

    Args:
        text: The text message to send to Google Chat

    Returns:
        Dict formatted as Workspace Add-on action response:
        {
            "hostAppDataAction": {
                "chatDataAction": {
                    "createMessageAction": {
                        "message": {"text": text}
                    }
                }
            }
        }

    References:
        - https://developers.google.com/workspace/add-ons/chat/send-messages
        - https://developers.google.com/workspace/add-ons/chat/convert
    """
    return {
        "hostAppDataAction": {
            "chatDataAction": {"createMessageAction": {"message": {"text": text}}}
        }
    }


def get_agent_name() -> str:
    """Dependency to inject agent name from server environment.

    Returns:
        The agent name from server configuration (e.g., "agent_foundation")
    """
    # Import here to avoid circular dependency
    from .server import env

    return env.agent_name


async def handle_chat_message(event: dict, agent_name: str) -> dict[str, Any]:
    """Handle Google Chat MESSAGE events with per-user session management.

    Framework-agnostic business logic that:
    1. Extracts user information from Google Chat event
    2. Gets or creates ADK session for the user
    3. Calls ADK agent via /run_sse endpoint with streaming
    4. Returns formatted response for Google Chat

    Uses SSE streaming to consume events from the ADK agent as they are generated,
    allowing for progress logging while staying within Chat's 30s timeout.

    Args:
        event: Google Chat event dict with Workspace Add-on structure:
               {"chat": {"user": {...}, "messagePayload": {"message": {...}, ...}}}
        agent_name: Name of the ADK agent (e.g., "dorothea")

    Returns:
        Dict in Workspace Add-on action format:
        {
            "hostAppDataAction": {
                "chatDataAction": {
                    "createMessageAction": {
                        "message": {"text": "response message"}
                    }
                }
            }
        }

    Raises:
        httpx.TimeoutException: If agent execution exceeds 30 seconds
        httpx.HTTPStatusError: If ADK endpoint returns error status
        Exception: For other errors during processing
    """
    # Build span attributes, excluding None values (OTel doesn't accept None)
    span_attributes = {"agent.name": agent_name}

    # Extract space info from Workspace Add-on format
    try:
        chat_data = event["chat"]
        payload = chat_data.get("messagePayload") or chat_data.get("appCommandPayload")
        if payload and "space" in payload:
            space = payload["space"]
            if space_name := space.get("name"):
                span_attributes["chat.space.name"] = space_name
            if space_type := space.get("type"):
                span_attributes["chat.space.type"] = space_type
    except (KeyError, TypeError):
        pass  # Space info is optional for observability

    with tracer.start_as_current_span(
        "handle_chat_message",
        attributes=span_attributes,
    ) as span:
        try:
            # Check for Workspace Add-on event format
            if not event.get("chat"):
                logger.warning("Invalid event: missing 'chat' field")
                span.set_attribute("chat.event.handled", False)
                span.set_attribute("chat.response.type", "error")
                return create_workspace_addon_response("Invalid event format")

            logger.info("Received Google Chat message event")

            # Extract message details from Workspace Add-on format
            chat_data = event["chat"]

            if "messagePayload" in chat_data:
                message_data = chat_data["messagePayload"].get("message", {})
            elif "appCommandPayload" in chat_data:
                message_data = chat_data["appCommandPayload"].get("message", {})
            else:
                # Handle first-time installation or events without message payload
                # (e.g., ADDED_TO_SPACE events)
                logger.info("Event without message payload, sending welcome message")
                chat_user_name = chat_data["user"]["name"]
                user_id = GoogleChatSessionManager.extract_user_id(chat_user_name)
                span.set_attribute("chat.user.id", user_id)
                span.set_attribute("chat.event.type", "welcome")
                return create_workspace_addon_response(
                    f"👋 Hi <users/{user_id}>! I'm Dorothea, your Google developer "
                    "documentation assistant. I have access to the Google Developer "
                    "Knowledge API to help you find accurate, up-to-date information "
                    "from official Google documentation.\n\n"
                    "Ask me anything about Google Cloud, Android, Chrome, Firebase, "
                    "TensorFlow, and more!"
                )

            user_message = message_data.get("text", "")
            chat_user_name = chat_data["user"]["name"]  # "users/123456789"
            user_display_name = chat_data["user"]["displayName"]

            # Add user context to span
            user_id = GoogleChatSessionManager.extract_user_id(chat_user_name)
            span.set_attribute("chat.user.id", user_id)
            span.set_attribute("chat.user.display_name", user_display_name)
            span.set_attribute("chat.message.length", len(user_message))

            logger.info(
                f"Processing message from {user_display_name} ({chat_user_name}): "
                f"{user_message[:50]}..."
            )

            # Get session service (configured with reasoning engine ID)
            from .server import env

            if not env.agent_engine:
                logger.error("AGENT_ENGINE not configured")
                error_msg = "AGENT_ENGINE not configured"
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                span.record_exception(ValueError(error_msg))
                return create_workspace_addon_response(
                    "Sorry, the agent is not properly configured for "
                    "session management."
                )

            # Session management with nested span
            with tracer.start_as_current_span("session.get_or_create") as session_span:
                session_service = create_session_service(
                    env.google_cloud_project,
                    env.google_cloud_location,
                    env.agent_engine,  # Full resource name; factory extracts ID
                )
                session_manager = GoogleChatSessionManager(session_service, agent_name)

                # Get or create session for this user
                session_id = await session_manager.get_or_create_session(chat_user_name)

                session_span.set_attribute("session.id", session_id)
                session_span.set_attribute("chat.user.id", user_id)

            logger.info(f"Using session {session_id} for user {user_id}")
            span.set_attribute("session.id", session_id)

            # ADK streaming call with nested span
            with tracer.start_as_current_span("adk.stream_response") as adk_span:
                adk_span.set_attribute("adk.endpoint", "http://localhost:8000/run_sse")
                adk_span.set_attribute("adk.agent_name", agent_name)

                # Call ADK's /run_sse endpoint with streaming
                async with (
                    httpx.AsyncClient() as client,
                    client.stream(
                        "POST",
                        "http://localhost:8000/run_sse",
                        json={
                            "app_name": agent_name,
                            "user_id": user_id,
                            "session_id": session_id,
                            "new_message": {"parts": [{"text": user_message}]},
                            "streaming": True,
                        },
                        timeout=30.0,
                    ) as response,
                ):
                    events = []
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            event_data = json.loads(line[6:])  # Skip "data: " prefix
                            events.append(event_data)

                            # Log progress for debugging
                            if event_data.get("type") == "message":
                                content = event_data.get("content", {})
                                parts = content.get("parts", [])
                                for part in parts:
                                    if part.get("text"):
                                        logger.debug(
                                            f"Agent progress: {part['text'][:50]}..."
                                        )

                    adk_span.set_attribute("adk.events.count", len(events))

            # Extract final response
            response_text = extract_agent_response(events)
            span.set_attribute("chat.response.length", len(response_text))
            span.set_attribute("chat.response.truncated", len(response_text) > 100)

            logger.info(f"Returning response: {response_text[:100]}...")
            response_dict = create_workspace_addon_response(response_text)
            return response_dict

        except httpx.TimeoutException as e:
            logger.error("Agent execution timeout", exc_info=True)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "timeout"))
            span.record_exception(e)
            return create_workspace_addon_response(
                "⏱️ Request timed out. Please try a simpler query or "
                "ask me to check fewer timecards."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Agent HTTP error: {e}", exc_info=True)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "http_error"))
            span.record_exception(e)
            span.set_attribute("http.status_code", e.response.status_code)
            return create_workspace_addon_response(
                "Sorry, the agent encountered an error. Please try again."
            )
        except Exception as e:
            logger.error(f"Chat webhook error: {e}", exc_info=True)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "unknown"))
            span.record_exception(e)
            return create_workspace_addon_response(
                "Sorry, I encountered an error processing your request."
            )


def extract_agent_response(events: list[dict]) -> str:
    """Extract final text response from ADK events.

    Processes the list of events returned from ADK's /run_sse endpoint and extracts
    the final text response. Searches events in reverse order to find the most recent
    message with text content.

    Args:
        events: List of event dictionaries from ADK /run_sse endpoint.
                Each event has structure: {"type": str, "content": {"parts": [...]}}

    Returns:
        The final text response from the agent, or a default message if no response
        was found.
    """
    # Events are list of event dicts from /run_sse
    for event in reversed(events):
        # Look for message content
        content = event.get("content")
        if content and isinstance(content, dict):
            parts = content.get("parts", [])
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if text and isinstance(text, str):
                            return text  # type: ignore[no-any-return]

    return "I processed your request but didn't generate a response."


async def handle_reset_command(event: dict, agent_name: str) -> dict[str, Any]:
    """Handle /reset command to clear user's conversation history.

    Deletes all ADK sessions for the Google Chat user, allowing them to start
    fresh conversations without historical context.

    Args:
        event: Google Chat event dict (Workspace Add-on format)
        agent_name: Name of the ADK agent (for logging)

    Returns:
        Dict in Workspace Add-on action format with success/error message
    """
    with tracer.start_as_current_span(
        "handle_reset_command",
        attributes={
            "chat.command": "reset",
            "agent.name": agent_name,
        },
    ) as span:
        # Extract user info from Workspace Add-on format
        chat_data = event["chat"]
        chat_user_name = chat_data["user"]["name"]
        user_display_name = chat_data["user"]["displayName"]
        user_id = GoogleChatSessionManager.extract_user_id(chat_user_name)

        span.set_attribute("chat.user.id", user_id)

        logger.info(f"Reset command from {user_display_name} ({chat_user_name})")

        # Get session service
        from .server import env

        if not env.agent_engine:
            logger.error("AGENT_ENGINE not configured")
            error_msg = "AGENT_ENGINE not configured"
            span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
            span.record_exception(ValueError(error_msg))
            return create_workspace_addon_response(
                "Sorry, the agent is not properly configured for session management."
            )

        try:
            with tracer.start_as_current_span("session.delete_all") as delete_span:
                session_service = create_session_service(
                    env.google_cloud_project,
                    env.google_cloud_location,
                    env.agent_engine,
                )
                session_manager = GoogleChatSessionManager(session_service, agent_name)

                # Delete all sessions for user
                count = await session_manager.delete_all_sessions(chat_user_name)

                delete_span.set_attribute("sessions.deleted_count", count)

            span.set_attribute("sessions.deleted_count", count)

            if count > 0:
                logger.info(f"Deleted {count} session(s) for user {chat_user_name}")
                return create_workspace_addon_response(
                    "✨ OK, let's start from the beginning! "
                    "Your conversation history has been reset."
                )
            else:
                logger.info(f"No sessions found for user {chat_user_name}")
                return create_workspace_addon_response(
                    "You don't have any active conversation history to reset."
                )

        except Exception as e:
            logger.error(f"Failed to reset sessions: {e}", exc_info=True)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "reset_failed"))
            span.record_exception(e)
            return create_workspace_addon_response(
                "Sorry, I encountered an error resetting your conversation."
            )


@chat_router.post("/webhook")
async def webhook(
    request: Request,
    agent_name: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Google Chat webhook endpoint for Workspace Add-on events.

    Receives Google Chat events via HTTP POST (Workspace Add-on format),
    detects /reset commands, and delegates to appropriate handlers.

    Args:
        request: FastAPI request object containing Google Chat event
        agent_name: Name of the ADK agent (injected via dependency)

    Returns:
        Dict in Workspace Add-on action format with response message
    """
    event = await request.json()

    logger.debug(f"Received Google Chat event: {event}")

    # Validate Workspace Add-on event format
    if not event.get("chat"):
        logger.warning("Invalid event: missing 'chat' field")
        return create_workspace_addon_response("Invalid event format")

    # Extract message text from Workspace Add-on format
    # If no message payload exists (e.g., first-time install), delegate to
    # handle_chat_message which will send welcome message
    chat_data = event["chat"]
    message_text = None

    if "messagePayload" in chat_data:
        with suppress(KeyError):
            message_text = chat_data["messagePayload"]["message"]["text"]
    elif "appCommandPayload" in chat_data:
        message_text = chat_data["appCommandPayload"].get("message", {}).get("text", "")

    # If we have message text, check for /reset command
    if message_text:
        message_text = message_text.strip().lower()
        logger.debug(f"Extracted message text: '{message_text}'")

        # Detect /reset command
        if message_text == "/reset":
            logger.debug("Detected /reset command")
            return await handle_reset_command(event, agent_name)

    # Regular message handling (includes welcome message for first-time users)
    logger.debug("Calling handle_chat_message")
    return await handle_chat_message(event, agent_name)
