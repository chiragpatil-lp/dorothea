"""Tests for Google Chat session management."""

from __future__ import annotations

from pytest_mock import MockerFixture

from dorothea.session import GoogleChatSessionManager, create_session_service


def test_create_session_service():
    """Test session service factory extracts reasoning engine ID."""
    service = create_session_service(
        "my-project",
        "us-central1",
        "projects/123/locations/us-central1/reasoningEngines/456",
    )
    assert service is not None


def test_extract_user_id():
    """Test extracting user ID from Google Chat user resource name."""
    assert GoogleChatSessionManager.extract_user_id("users/123456789") == "123456789"
    assert GoogleChatSessionManager.extract_user_id("users/abc") == "abc"


async def test_list_sessions_with_sessions(
    mock_session_service,
    mock_session,
    mock_list_sessions_response,
    mocker: MockerFixture,
):
    """Test listing sessions when user has sessions."""
    sessions = [mock_session("session-abc-123"), mock_session("session-xyz-456")]
    response = mock_list_sessions_response(sessions)

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=response)

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    sessions = await manager.list_sessions("users/123456789")

    assert sessions == ["session-abc-123", "session-xyz-456"]
    mock_session_service.list_sessions.assert_called_once_with(
        app_name="dorothea",
        user_id="123456789",
    )


async def test_list_sessions_no_sessions(
    mock_session_service, mock_list_sessions_response, mocker: MockerFixture
):
    """Test listing sessions when user has no sessions."""
    response = mock_list_sessions_response([])

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=response)

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    sessions = await manager.list_sessions("users/123456789")

    assert sessions == []


async def test_create_session(
    mock_session_service, mock_session, mocker: MockerFixture
):
    """Test creating new session for user."""
    session = mock_session("session-new-123")

    mock_session_service.create_session = mocker.AsyncMock(return_value=session)

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    session_id = await manager.create_session("users/123456789")

    assert session_id == "session-new-123"
    mock_session_service.create_session.assert_called_once_with(
        app_name="dorothea",
        user_id="123456789",
    )


async def test_get_or_create_session_existing(
    mock_session_service,
    mock_session,
    mock_list_sessions_response,
    mocker: MockerFixture,
    mock_tracer,
):
    """Test get-or-create when session exists."""
    session = mock_session("session-existing-123")
    response = mock_list_sessions_response([session])

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=response)

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    session_id = await manager.get_or_create_session("users/123456789")

    assert session_id == "session-existing-123"
    mock_session_service.create_session.assert_not_called()

    # Verify span attributes for existing session
    assert len(mock_tracer.spans) == 1
    span_name, span = mock_tracer.spans[0]
    assert span_name == "session.get_or_create"
    assert span.attributes["chat.user.name"] == "users/123456789"
    assert span.attributes["chat.user.id"] == "123456789"
    assert span.attributes["session.created"] is False
    assert span.attributes["session.id"] == "session-existing-123"


async def test_get_or_create_session_new(
    mock_session_service,
    mock_session,
    mock_list_sessions_response,
    mocker: MockerFixture,
    mock_tracer,
):
    """Test get-or-create when no session exists."""
    empty_response = mock_list_sessions_response([])
    new_session = mock_session("session-new-123")

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=empty_response)
    mock_session_service.create_session = mocker.AsyncMock(return_value=new_session)

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    session_id = await manager.get_or_create_session("users/123456789")

    assert session_id == "session-new-123"
    mock_session_service.create_session.assert_called_once()

    # Verify span attributes for new session
    assert len(mock_tracer.spans) == 1
    span_name, span = mock_tracer.spans[0]
    assert span_name == "session.get_or_create"
    assert span.attributes["chat.user.id"] == "123456789"
    assert span.attributes["session.created"] is True
    assert span.attributes["session.id"] == "session-new-123"


async def test_delete_session(mock_session_service, mocker: MockerFixture):
    """Test deleting specific session."""
    mock_session_service.delete_session = mocker.AsyncMock()

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    await manager.delete_session("users/123456789", "session-abc-123")

    mock_session_service.delete_session.assert_called_once_with(
        app_name="dorothea",
        user_id="123456789",
        session_id="session-abc-123",
    )


async def test_delete_all_sessions_with_sessions(
    mock_session_service,
    mock_session,
    mock_list_sessions_response,
    mocker: MockerFixture,
    mock_tracer,
):
    """Test deleting all sessions when user has sessions."""
    sessions = [mock_session("session-1"), mock_session("session-2")]
    response = mock_list_sessions_response(sessions)

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=response)
    mock_session_service.delete_session = mocker.AsyncMock()

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    count = await manager.delete_all_sessions("users/123456789")

    assert count == 2
    assert mock_session_service.delete_session.call_count == 2

    # Verify span creation and attributes
    span_names = [name for name, _ in mock_tracer.spans]
    assert "session.delete_all" in span_names
    assert span_names.count("session.delete") == 2

    # Verify delete_all span attributes
    for name, span in mock_tracer.spans:
        if name == "session.delete_all":
            assert span.attributes["chat.user.name"] == "users/123456789"
            assert span.attributes["chat.user.id"] == "123456789"
            assert span.attributes["sessions.deleted_count"] == 2
            assert span.attributes["sessions.total_count"] == 2

    # Verify individual delete spans have session IDs
    delete_span_ids = []
    for name, span in mock_tracer.spans:
        if name == "session.delete":
            delete_span_ids.append(span.attributes["session.id"])
    assert "session-1" in delete_span_ids
    assert "session-2" in delete_span_ids


async def test_delete_all_sessions_no_sessions(
    mock_session_service,
    mock_list_sessions_response,
    mocker: MockerFixture,
    mock_tracer,
):
    """Test deleting all sessions when user has no sessions."""
    response = mock_list_sessions_response([])

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=response)
    mock_session_service.delete_session = mocker.AsyncMock()

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    count = await manager.delete_all_sessions("users/123456789")

    assert count == 0
    mock_session_service.delete_session.assert_not_called()

    # Verify span attributes when no sessions to delete
    assert len(mock_tracer.spans) == 1
    span_name, span = mock_tracer.spans[0]
    assert span_name == "session.delete_all"
    assert span.attributes["sessions.deleted_count"] == 0
    assert span.attributes["sessions.total_count"] == 0


async def test_delete_all_sessions_partial_failure(
    mock_session_service,
    mock_session,
    mock_list_sessions_response,
    mocker: MockerFixture,
    mock_tracer,
):
    """Test deleting sessions when some deletions fail."""
    sessions = [mock_session("session-1"), mock_session("session-2")]
    response = mock_list_sessions_response(sessions)

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=response)

    # First delete succeeds, second fails
    mock_session_service.delete_session = mocker.AsyncMock(
        side_effect=[None, RuntimeError("Delete failed")]
    )

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    count = await manager.delete_all_sessions("users/123456789")

    assert count == 1  # Only one deletion succeeded
    assert mock_session_service.delete_session.call_count == 2

    # Verify exception was recorded in nested span
    delete_spans = [
        span for name, span in mock_tracer.spans if name == "session.delete"
    ]
    assert len(delete_spans) == 2

    # Second delete span should have exception
    assert len(delete_spans[1].exceptions) == 1
    assert isinstance(delete_spans[1].exceptions[0], RuntimeError)
    assert delete_spans[1].status is not None


async def test_delete_all_sessions_span_scoping(
    mock_session_service,
    mock_session,
    mock_list_sessions_response,
    mocker: MockerFixture,
    mock_tracer,
):
    """Test that span is only accessed within context manager scope.

    Regression test for bug where delete_span was accessed in except block
    outside the context manager scope, violating proper context manager usage.

    MockTracer enforces that spans are only accessed within their context
    manager scope. Would raise RuntimeError if the old buggy pattern
    (try wrapping with) was used instead of the fixed pattern (try inside with).
    """
    # Setup mocks
    sessions = [mock_session("session-1"), mock_session("session-2")]
    response = mock_list_sessions_response(sessions)

    mock_session_service.list_sessions = mocker.AsyncMock(return_value=response)

    # First delete succeeds, second fails
    mock_session_service.delete_session = mocker.AsyncMock(
        side_effect=[None, RuntimeError("Delete failed")]
    )

    manager = GoogleChatSessionManager(mock_session_service, "dorothea")
    count = await manager.delete_all_sessions("users/123456789")

    # Should succeed without RuntimeError about accessing span after exit
    assert count == 1
    assert len(mock_tracer.spans) == 3  # parent + 2 delete spans

    # Verify second span has exception recorded
    delete_spans = [
        span for name, span in mock_tracer.spans if name == "session.delete"
    ]
    assert len(delete_spans) == 2
    assert len(delete_spans[1].exceptions) == 1
    assert isinstance(delete_spans[1].exceptions[0], RuntimeError)
