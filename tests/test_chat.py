"""Unit tests for Google Chat integration.

Tests the framework-agnostic business logic in the chat module without requiring
a running FastAPI app or ADK agent. Uses mocks to simulate HTTP responses from
the ADK /run_sse endpoint.
"""

from __future__ import annotations

from typing import Any

import httpx
from pytest_mock import MockerFixture

from dorothea.chat import (
    extract_agent_response,
    handle_chat_message,
    handle_reset_command,
)


async def test_handle_chat_message_success(
    chat_message_event: dict[str, Any],
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test successful MESSAGE event processing."""
    sse_lines = [
        'data: {"type": "message", "content": {"parts": [{"text": "Querying..."}]}}',
        'data: {"type": "message", "content": {"parts": [{"text": "Found 3."}]}}',
        'data: {"type": "end"}',
    ]

    mock_create_session_service(session_id="test-session-success")
    mock_httpx_client(sse_lines=sse_lines, exception=None)

    result = await handle_chat_message(chat_message_event, agent_name="dorothea")

    assert result == {"actionResponse": {"type": "NEW_MESSAGE"}, "text": "Found 3."}

    # Verify span creation
    span_names = [name for name, _ in mock_tracer.spans]
    assert "handle_chat_message" in span_names
    assert "session.get_or_create" in span_names
    assert "adk.stream_response" in span_names


async def test_handle_chat_message_minimal_event(
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test MESSAGE event with minimal/missing optional fields."""
    # Event with type=MESSAGE, space with name but no type
    minimal_event = {
        "type": "MESSAGE",
        "message": {"text": "Hello"},
        "user": {"name": "users/TEST_USER", "displayName": "Test User"},
        "space": {"name": "spaces/test-space"},  # Has name but no type
    }

    sse_lines = [
        'data: {"type": "message", "content": {"parts": [{"text": "Response"}]}}',
    ]

    mock_create_session_service(session_id="test-session-minimal")
    mock_httpx_client(sse_lines=sse_lines, exception=None)

    result = await handle_chat_message(minimal_event, agent_name="dorothea")

    assert result == {"actionResponse": {"type": "NEW_MESSAGE"}, "text": "Response"}

    # Verify span attributes set correctly even when space.type is missing
    for name, span in mock_tracer.spans:
        if name == "handle_chat_message":
            assert "chat.space.name" in span.attributes
            assert "chat.space.type" not in span.attributes  # Missing in event


async def test_handle_chat_message_all_optional_fields_missing(
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test MESSAGE event with all optional span attribute fields missing."""
    # Event with no type, no space, triggering all branch coverage gaps
    minimal_event = {
        "message": {"text": "Hello"},
        "user": {"name": "users/TEST_USER", "displayName": "Test User"},
        # No type field, no space field
    }

    sse_lines = [
        'data: {"type": "message", "content": {"parts": [{"text": "Response"}]}}',
    ]

    mock_create_session_service(session_id="test-session-all-missing")
    mock_httpx_client(sse_lines=sse_lines, exception=None)

    # Event gets "Event received" response because no type field
    result = await handle_chat_message(minimal_event, agent_name="dorothea")

    assert result == {"actionResponse": {"type": "NEW_MESSAGE"}, "text": "Event received"}

    # Verify span was created with only agent.name attribute
    for name, span in mock_tracer.spans:
        if name == "handle_chat_message":
            assert span.attributes.get("agent.name") == "dorothea"
            assert "chat.event.type" not in span.attributes
            assert "chat.space.name" not in span.attributes
            assert "chat.space.type" not in span.attributes


async def test_handle_chat_message_space_without_name(
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test MESSAGE event with space that has no name field."""
    # Event with space object but no name field (triggers 81->83 branch)
    event_space_no_name = {
        "type": "MESSAGE",
        "message": {"text": "Hello"},
        "user": {"name": "users/TEST_USER", "displayName": "Test User"},
        "space": {"type": "DM"},  # Has type but no name
    }

    sse_lines = [
        'data: {"type": "message", "content": {"parts": [{"text": "Response"}]}}',
    ]

    mock_create_session_service(session_id="test-session-no-space-name")
    mock_httpx_client(sse_lines=sse_lines, exception=None)

    result = await handle_chat_message(event_space_no_name, agent_name="dorothea")

    assert result == {"actionResponse": {"type": "NEW_MESSAGE"}, "text": "Response"}

    # Verify space.type was set but not space.name
    for name, span in mock_tracer.spans:
        if name == "handle_chat_message":
            assert "chat.space.name" not in span.attributes
            assert span.attributes.get("chat.space.type") == "DM"


async def test_handle_chat_message_non_message_event(
    mock_tracer,
) -> None:
    """Test handling non-MESSAGE events (should return acknowledgment)."""
    event = {"type": "ADDED_TO_SPACE", "space": {"name": "spaces/TEST"}}

    result = await handle_chat_message(event, agent_name="dorothea")

    assert result == {"actionResponse": {"type": "NEW_MESSAGE"}, "text": "Event received"}

    # Verify span attributes for non-MESSAGE event
    assert len(mock_tracer.spans) == 1
    span_name, span = mock_tracer.spans[0]
    assert span_name == "handle_chat_message"
    assert span.attributes["chat.event.type"] == "ADDED_TO_SPACE"
    assert span.attributes["chat.event.handled"] is False
    assert span.attributes["chat.response.type"] == "acknowledgment"


async def test_handle_chat_message_timeout(
    simple_message_event: dict[str, Any],
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test timeout handling when ADK takes too long."""
    mock_create_session_service(session_id="test-session-timeout")
    mock_httpx_client(sse_lines=None, exception=httpx.TimeoutException("Timeout"))

    result = await handle_chat_message(simple_message_event, agent_name="dorothea")

    assert "timed out" in result["text"].lower()

    # Verify error was recorded in span
    for name, span in mock_tracer.spans:
        if name == "handle_chat_message":
            assert len(span.exceptions) == 1
            assert isinstance(span.exceptions[0], httpx.TimeoutException)
            assert span.status is not None  # Status was set


async def test_handle_chat_message_http_error(
    simple_message_event: dict[str, Any],
    mock_httpx_client,
    mock_create_session_service,
    mocker: MockerFixture,
    mock_tracer,
) -> None:
    """Test handling HTTP errors from ADK endpoint."""
    mock_create_session_service(session_id="test-session-http-error")
    http_error = httpx.HTTPStatusError(
        "Internal Server Error",
        request=mocker.Mock(),
        response=mocker.Mock(status_code=500),
    )
    mock_httpx_client(sse_lines=None, exception=http_error)

    result = await handle_chat_message(simple_message_event, agent_name="dorothea")

    assert "error" in result["text"].lower()

    # Verify error was recorded with HTTP status code
    for name, span in mock_tracer.spans:
        if name == "handle_chat_message":
            assert len(span.exceptions) == 1
            assert isinstance(span.exceptions[0], httpx.HTTPStatusError)
            assert span.attributes["http.status_code"] == 500


async def test_handle_chat_message_generic_exception(
    simple_message_event: dict[str, Any],
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test handling unexpected exceptions."""
    mock_create_session_service(session_id="test-session-exception")
    mock_httpx_client(sse_lines=None, exception=Exception("Unexpected error"))

    result = await handle_chat_message(simple_message_event, agent_name="dorothea")

    assert "error" in result["text"].lower()

    # Verify generic exception was recorded
    for name, span in mock_tracer.spans:
        if name == "handle_chat_message":
            assert len(span.exceptions) == 1
            assert span.exceptions[0].args[0] == "Unexpected error"


async def test_handle_chat_message_no_agent_engine(
    simple_message_event: dict[str, Any],
    mocker: MockerFixture,
    mock_tracer,
) -> None:
    """Test handling MESSAGE when AGENT_ENGINE is not configured."""
    # Mock server.env.agent_engine to be None
    # (patch in server module where it's defined)
    mocker.patch("dorothea.server.env.agent_engine", None)

    result = await handle_chat_message(simple_message_event, agent_name="dorothea")

    assert "not properly configured" in result["text"]

    # Verify error status and exception recording
    for name, span in mock_tracer.spans:
        if name == "handle_chat_message":
            assert len(span.exceptions) == 1
            assert isinstance(span.exceptions[0], ValueError)
            assert span.status is not None


async def test_handle_chat_message_with_non_data_sse_lines(
    chat_message_event: dict[str, Any],
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test handling SSE stream with non-data lines (comments, etc)."""
    sse_lines = [
        ": this is a comment",
        'data: {"type": "message", "content": {"parts": [{"text": "Response"}]}}',
        "event: custom-event",
        'data: {"type": "end"}',
    ]

    mock_create_session_service(session_id="test-session-non-data")
    mock_httpx_client(sse_lines=sse_lines, exception=None)

    result = await handle_chat_message(chat_message_event, agent_name="dorothea")

    assert result == {"actionResponse": {"type": "NEW_MESSAGE"}, "text": "Response"}

    # Verify spans were created
    span_names = [name for name, _ in mock_tracer.spans]
    assert "handle_chat_message" in span_names
    assert "adk.stream_response" in span_names


async def test_handle_chat_message_with_parts_without_text(
    chat_message_event: dict[str, Any],
    mock_httpx_client,
    mock_create_session_service,
    mock_tracer,
) -> None:
    """Test handling events with parts that don't have text."""
    sse_lines = [
        'data: {"type": "message", "content": {"parts": [{"image": "data:..."}]}}',
        'data: {"type": "message", "content": {"parts": [{"text": "Final"}]}}',
        'data: {"type": "end"}',
    ]

    mock_create_session_service(session_id="test-session-no-text-parts")
    mock_httpx_client(sse_lines=sse_lines, exception=None)

    result = await handle_chat_message(chat_message_event, agent_name="dorothea")

    assert result == {"actionResponse": {"type": "NEW_MESSAGE"}, "text": "Final"}

    # Verify event count in ADK span
    for name, span in mock_tracer.spans:
        if name == "adk.stream_response":
            assert span.attributes["adk.events.count"] == 3


def test_extract_agent_response_with_text() -> None:
    """Test extracting text response from ADK events."""
    events: list[dict[str, Any]] = [
        {
            "type": "message",
            "content": {"parts": [{"text": "Searching for timecards..."}]},
        },
        {"type": "message", "content": {"parts": [{"text": "Found 3 timecards."}]}},
        {"type": "end"},
    ]

    result = extract_agent_response(events)

    # Should return last message with text
    assert result == "Found 3 timecards."


def test_extract_agent_response_no_text() -> None:
    """Test extracting response when no text content in events."""
    events: list[dict[str, Any]] = [
        {"type": "tool_call", "content": {}},
        {"type": "end"},
    ]

    result = extract_agent_response(events)

    assert "didn't generate a response" in result


def test_extract_agent_response_empty_events() -> None:
    """Test extracting response from empty event list."""
    result = extract_agent_response([])

    assert "didn't generate a response" in result


def test_extract_agent_response_malformed_events() -> None:
    """Test extracting response from malformed events."""
    events: list[dict[str, Any]] = [
        {"type": "message"},  # No content
        {"type": "message", "content": {}},  # No parts
        {"type": "message", "content": {"parts": []}},  # Empty parts
        {"type": "message", "content": {"parts": [{"no_text": "here"}]}},  # No text
        {"type": "message", "content": {"parts": "not-a-list"}},  # Parts not a list
        {"type": "message", "content": {"parts": ["not-a-dict"]}},  # Part not a dict
    ]

    result = extract_agent_response(events)

    assert "didn't generate a response" in result


def test_extract_agent_response_multiple_parts() -> None:
    """Test extracting response when event has multiple parts."""
    events: list[dict[str, Any]] = [
        {
            "type": "message",
            "content": {
                "parts": [
                    {"image": "data:..."},  # Non-text part
                    {"text": "This is the text response"},  # Text part
                ]
            },
        }
    ]

    result = extract_agent_response(events)

    assert result == "This is the text response"


def test_extract_agent_response_reverse_order() -> None:
    """Test that extraction uses last message (reverse order)."""
    events: list[dict[str, Any]] = [
        {"type": "message", "content": {"parts": [{"text": "First message"}]}},
        {"type": "message", "content": {"parts": [{"text": "Second message"}]}},
        {"type": "message", "content": {"parts": [{"text": "Third message"}]}},
    ]

    result = extract_agent_response(events)

    # Should return the last message
    assert result == "Third message"


async def test_handle_reset_command_success(
    reset_command_event: dict[str, Any],
    mocker: MockerFixture,
    mock_tracer,
) -> None:
    """Test /reset command successfully deletes sessions."""
    # Create mock that simulates deleting 1 session
    mock_session = mocker.Mock()
    mock_session.id = "session-to-delete"
    mock_list_result = mocker.Mock()
    mock_list_result.sessions = [mock_session]

    mock_service = mocker.Mock()
    mock_service.list_sessions = mocker.AsyncMock(return_value=mock_list_result)
    mock_service.delete_session = mocker.AsyncMock()

    mocker.patch("dorothea.chat.create_session_service", return_value=mock_service)

    result = await handle_reset_command(reset_command_event, agent_name="dorothea")

    assert "start from the beginning" in result["text"]
    assert "reset" in result["text"].lower()
    mock_service.delete_session.assert_called_once()

    # Verify span creation and attributes
    span_names = [name for name, _ in mock_tracer.spans]
    assert "handle_reset_command" in span_names
    assert "session.delete_all" in span_names

    for name, span in mock_tracer.spans:
        if name == "handle_reset_command":
            assert span.attributes["chat.command"] == "reset"
            assert span.attributes["agent.name"] == "dorothea"
            assert span.attributes["chat.user.id"] == "TEST_USER"
            assert span.attributes["sessions.deleted_count"] == 1


async def test_handle_reset_command_no_sessions(
    reset_command_event: dict[str, Any],
    mocker: MockerFixture,
    mock_tracer,
) -> None:
    """Test /reset command when user has no sessions."""
    # Create mock that simulates no sessions
    mock_list_result = mocker.Mock()
    mock_list_result.sessions = []

    mock_service = mocker.Mock()
    mock_service.list_sessions = mocker.AsyncMock(return_value=mock_list_result)

    mocker.patch("dorothea.chat.create_session_service", return_value=mock_service)

    result = await handle_reset_command(reset_command_event, agent_name="dorothea")

    assert "don't have any active conversation" in result["text"]

    # Verify deleted count is 0
    for name, span in mock_tracer.spans:
        if name == "handle_reset_command":
            assert span.attributes["sessions.deleted_count"] == 0


async def test_handle_reset_command_no_agent_engine(
    reset_command_event: dict[str, Any],
    mocker: MockerFixture,
    mock_tracer,
) -> None:
    """Test /reset command when AGENT_ENGINE is not configured."""
    # Mock server.env.agent_engine to be None
    # (patch in server module where it's defined)
    mocker.patch("dorothea.server.env.agent_engine", None)

    result = await handle_reset_command(reset_command_event, agent_name="dorothea")

    assert "not properly configured" in result["text"]

    # Verify error was recorded
    for name, span in mock_tracer.spans:
        if name == "handle_reset_command":
            assert len(span.exceptions) == 1
            assert isinstance(span.exceptions[0], ValueError)
            assert span.status is not None


async def test_handle_reset_command_generic_exception(
    reset_command_event: dict[str, Any],
    mocker: MockerFixture,
    mock_tracer,
) -> None:
    """Test /reset command when a generic exception occurs."""
    # Mock create_session_service to raise a generic exception
    mocker.patch(
        "dorothea.chat.create_session_service",
        side_effect=RuntimeError("Unexpected error"),
    )

    result = await handle_reset_command(reset_command_event, agent_name="dorothea")

    assert "encountered an error" in result["text"]

    # Verify exception was recorded
    for name, span in mock_tracer.spans:
        if name == "handle_reset_command":
            assert len(span.exceptions) == 1
            assert isinstance(span.exceptions[0], RuntimeError)
            assert span.status is not None
