"""Integration tests for Google Chat webhook route.

Tests the FastAPI route wrapper and dependency injection without requiring
the full ADK agent to be running. Uses TestClient to test the actual HTTP layer.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from dorothea.chat import chat_router, get_agent_name


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI app with chat router for testing."""
    test_app = FastAPI()
    test_app.include_router(chat_router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_get_agent_name(mocker: MockerFixture) -> None:
    """Test get_agent_name dependency function."""
    mock_env = mocker.patch("dorothea.server.env")
    mock_env.agent_name = "test-agent"

    result = get_agent_name()

    assert result == "test-agent"


def test_webhook_route_success(
    client: TestClient, chat_message_event: dict[str, Any], mocker: MockerFixture
) -> None:
    """Test webhook route with successful MESSAGE event."""
    # Mock the handle_chat_message function
    mock_handle = mocker.patch(
        "dorothea.chat.handle_chat_message",
        return_value={
            "actionResponse": {"type": "NEW_MESSAGE"},
            "text": "Success response",
        },
    )

    response = client.post("/chat/webhook", json=chat_message_event)

    assert response.status_code == 200
    assert response.json() == {
        "actionResponse": {"type": "NEW_MESSAGE"},
        "text": "Success response",
    }
    mock_handle.assert_called_once()


def test_webhook_route_reset_command(
    client: TestClient,
    reset_command_event: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    """Test webhook route with /reset command."""
    # Mock handle_reset_command
    mock_reset = mocker.patch(
        "dorothea.chat.handle_reset_command",
        return_value={
            "actionResponse": {"type": "NEW_MESSAGE"},
            "text": "Conversation reset",
        },
    )

    response = client.post("/chat/webhook", json=reset_command_event)

    assert response.status_code == 200
    assert response.json() == {
        "actionResponse": {"type": "NEW_MESSAGE"},
        "text": "Conversation reset",
    }
    mock_reset.assert_called_once()


def test_webhook_route_reset_command_case_insensitive(
    client: TestClient, mocker: MockerFixture
) -> None:
    """Test webhook route handles /reset in various cases."""
    reset_event = {
        "type": "MESSAGE",
        "message": {"text": "  /RESET  "},  # Uppercase with spaces
        "user": {"name": "users/TEST_USER", "displayName": "Test User"},
        "space": {"name": "spaces/TEST", "type": "DM"},
    }

    # Mock handle_reset_command
    mock_reset = mocker.patch(
        "dorothea.chat.handle_reset_command",
        return_value={
            "actionResponse": {"type": "NEW_MESSAGE"},
            "text": "Conversation reset",
        },
    )

    response = client.post("/chat/webhook", json=reset_event)

    assert response.status_code == 200
    assert response.json() == {
        "actionResponse": {"type": "NEW_MESSAGE"},
        "text": "Conversation reset",
    }
    mock_reset.assert_called_once()


def test_webhook_route_non_message_event(client: TestClient) -> None:
    """Test webhook route with non-MESSAGE event (no mocks for full coverage)."""
    event = {"type": "ADDED_TO_SPACE", "space": {"name": "spaces/TEST"}}

    response = client.post("/chat/webhook", json=event)

    assert response.status_code == 200
    assert response.json() == {
        "actionResponse": {"type": "NEW_MESSAGE"},
        "text": "Event received",
    }
