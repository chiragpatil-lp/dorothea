"""Shared pytest fixtures for all tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pytest_mock import MockerFixture, MockType


def pytest_configure(config: pytest.Config) -> None:
    """Pytest hook to set up environment before test collection.

    IMPORTANT: This hook runs BEFORE pytest's plugin system is fully initialized,
    including pytest-mock. We must use unittest.mock.patch here because:

    1. This hook runs before test collection
    2. Test collection imports test modules, which imports agent_foundation modules
    3. Agent modules may trigger API calls during import (auth, config loading)
    4. pytest-mock's mocker/session_mocker fixtures aren't available until AFTER
       test collection completes

    Pytest execution order:
    - pytest_configure() ← We are here (only stdlib available)
    - Test collection (imports happen, triggers API calls if not mocked)
    - Session setup (fixtures become available)
    - Test execution

    Therefore, unittest.mock is the ONLY tool available at this stage. This is
    the only place in the codebase where unittest.mock is used - all other mocking
    (fixtures and tests) uses pytest-mock's mocker fixture.
    """
    from unittest.mock import Mock, patch

    # Patch load_dotenv to prevent loading real .env file during module imports
    load_dotenv_patcher = patch("dotenv.load_dotenv")
    load_dotenv_patcher.start()

    # Patch google.auth.default to prevent Application Default Credentials lookup
    mock_credentials = Mock()
    mock_credentials.token = "test-mock-token-totally-not-real"  # noqa: S105
    mock_credentials.valid = True
    mock_credentials.expired = False
    mock_credentials.refresh = Mock()
    mock_credentials.universe_domain = "googleapis.com"

    # Patch both public and private auth paths (ADK uses private path internally)
    auth_patcher = patch(
        "google.auth.default", return_value=(mock_credentials, "test-project")
    )
    auth_patcher.start()

    auth_private_patcher = patch(
        "google.auth._default.default", return_value=(mock_credentials, "test-project")
    )
    auth_private_patcher.start()

    # Set test environment variables before any imports occur
    # Use direct assignment (not setdefault) since we're preventing .env loading
    import os

    os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
    os.environ["AGENT_NAME"] = "test-agent"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"


# ADK Callback Mock Objects for testing callbacks
class MockState:
    """Mock State object for ADK callback testing.

    Supports both dictionary-style access and to_dict() method
    to match ADK's state interface.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize mock state with optional data."""
        self._data = data if data is not None else {}

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from state with optional default."""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get item using dictionary syntax."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item using dictionary syntax."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self._data


class MockContent:
    """Mock Content object for ADK callback testing.

    Used for user_content and llm_content in callbacks.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize mock content with optional data."""
        self._data = data if data is not None else {"text": "test content"}

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        """Serialize content to dictionary."""
        return self._data


class MockSession:
    """Mock ADK Session for testing.

    Minimal mock used by MockReadonlyContext.
    """

    def __init__(self, user_id: str = "test_user_123") -> None:
        """Initialize mock session with user_id."""
        self.user_id = user_id


class MockMemoryCallbackContext:
    """Minimal mock CallbackContext for add_session_to_memory callback testing.

    Controls behavior through constructor parameters instead of rebuilding
    ADK's internal logic. This keeps tests independent of ADK implementation.
    """

    def __init__(
        self,
        should_raise: type[Exception] | None = None,
        error_message: str = "",
    ) -> None:
        """Initialize mock callback context with controlled behavior.

        Args:
            should_raise: Exception type to raise when add_session_to_memory is called.
                         None means the call succeeds.
            error_message: Message for the exception if should_raise is set.
        """
        self._should_raise = should_raise
        self._error_message = error_message
        self.add_session_to_memory_called = False

    async def add_session_to_memory(self) -> None:
        """Mock implementation that either succeeds or raises controlled exception.

        Raises:
            Exception: The exception type configured in __init__ if should_raise is set.
        """
        self.add_session_to_memory_called = True
        if self._should_raise:
            raise self._should_raise(self._error_message)


class MockLoggingCallbackContext:
    """Mock CallbackContext for LoggingCallbacks testing.

    Used for agent and model callbacks testing.
    """

    def __init__(
        self,
        agent_name: str = "test_agent",
        invocation_id: str = "test-invocation-123",
        state: MockState | None = None,
        user_content: MockContent | None = None,
    ) -> None:
        """Initialize mock callback context for logging callbacks."""
        self.agent_name = agent_name
        self.invocation_id = invocation_id
        self.state = state if state is not None else MockState()
        self.user_content = user_content


class MockLlmRequest:
    """Mock LlmRequest for model callbacks."""

    def __init__(self, contents: list[MockContent] | None = None) -> None:
        """Initialize mock LLM request."""
        if contents is None:
            contents = [
                MockContent({"text": "system prompt"}),
                MockContent({"text": "user message"}),
            ]
        self.contents = contents


class MockLlmResponse:
    """Mock LlmResponse for model callbacks."""

    def __init__(self, content: MockContent | None = None) -> None:
        """Initialize mock LLM response."""
        self.content = content


class MockEventActions:
    """Mock EventActions for tool callbacks."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize mock event actions."""
        self._data = data if data is not None else {"action": "execute"}

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        """Serialize actions to dictionary."""
        return self._data


class MockToolContext:
    """Mock ToolContext for tool callbacks."""

    def __init__(
        self,
        agent_name: str = "test_agent",
        invocation_id: str = "test-invocation-456",
        state: MockState | None = None,
        user_content: MockContent | None = None,
        actions: MockEventActions | None = None,
    ) -> None:
        """Initialize mock tool context."""
        self.agent_name = agent_name
        self.invocation_id = invocation_id
        self.state = state if state is not None else MockState()
        self.user_content = user_content
        self.actions = actions if actions is not None else MockEventActions()


class MockBaseTool:
    """Mock BaseTool for tool callbacks."""

    def __init__(self, name: str = "test_tool") -> None:
        """Initialize mock tool."""
        self.name = name


class MockReadonlyContext:
    """Mock ReadonlyContext for testing InstructionProvider functions.

    Provides read-only access to invocation metadata and session state,
    matching the interface of google.adk.agents.readonly_context.ReadonlyContext.

    To customize user_id, pass a MockSession with the desired user_id:
        MockReadonlyContext(session=MockSession(user_id="custom_user"))
    """

    def __init__(
        self,
        agent_name: str = "test_agent",
        invocation_id: str = "test-inv-readonly",
        state: dict[str, Any] | None = None,
        user_content: MockContent | None = None,
        session: MockSession | None = None,
    ) -> None:
        """Initialize mock readonly context.

        Args:
            agent_name: Name of the agent.
            invocation_id: ID of the current invocation.
            state: Session state dictionary (read-only).
            user_content: Optional user content that started the invocation.
            session: Optional session object. If not provided, creates MockSession
                     with default user_id.
        """
        self._agent_name = agent_name
        self._invocation_id = invocation_id
        self._state = state if state is not None else {}
        self._user_content = user_content
        self._session = session if session is not None else MockSession()

    @property
    def agent_name(self) -> str:
        """The name of the agent that is currently running."""
        return self._agent_name

    @property
    def invocation_id(self) -> str:
        """The current invocation id."""
        return self._invocation_id

    @property
    def state(self) -> dict[str, Any]:
        """The state of the current session (read-only)."""
        return self._state.copy()  # Return a copy to enforce read-only

    @property
    def user_content(self) -> MockContent | None:
        """The user content that started this invocation."""
        return self._user_content

    @property
    def session(self) -> MockSession:
        """The current session for this invocation."""
        return self._session

    @property
    def user_id(self) -> str:
        """The user ID from the current session."""
        return self._session.user_id


# Fixtures for ADK callback testing
@pytest.fixture
def mock_state() -> MockState:
    """Create a mock state with test data."""
    return MockState({"user_id": "user123", "session_data": {"key": "value"}})


@pytest.fixture
def mock_content() -> MockContent:
    """Create a mock content with test data."""
    return MockContent({"text": "Hello, agent!"})


@pytest.fixture
def mock_logging_callback_context(
    mock_state: MockState, mock_content: MockContent
) -> MockLoggingCallbackContext:
    """Create a mock logging callback context with full data."""
    return MockLoggingCallbackContext(
        agent_name="my_agent",
        invocation_id="inv-789",
        state=mock_state,
        user_content=mock_content,
    )


@pytest.fixture
def mock_llm_request() -> MockLlmRequest:
    """Create a mock LLM request with default messages."""
    return MockLlmRequest(
        contents=[
            MockContent({"text": "system prompt"}),
            MockContent({"text": "user message"}),
        ]
    )


@pytest.fixture
def mock_llm_response() -> MockLlmResponse:
    """Create a mock LLM response with content."""
    return MockLlmResponse(
        content=MockContent({"text": "The answer is 42", "confidence": 0.95})
    )


@pytest.fixture
def mock_event_actions() -> MockEventActions:
    """Create mock event actions with test data."""
    return MockEventActions({"action": "run", "params": ["arg1", "arg2"]})


@pytest.fixture
def mock_tool_context(
    mock_event_actions: MockEventActions,
) -> MockToolContext:
    """Create a mock tool context with full data."""
    return MockToolContext(
        agent_name="tool_agent",
        invocation_id="tool-inv-123",
        state=MockState({"tool_state": "active"}),
        user_content=MockContent({"text": "Execute tool"}),
        actions=mock_event_actions,
    )


@pytest.fixture
def mock_tool_context_empty_state(
    mock_event_actions: MockEventActions,
) -> MockToolContext:
    """Create a mock tool context with empty state."""
    return MockToolContext(
        agent_name="tool_agent",
        invocation_id="tool-inv-empty",
        state=MockState({}),
        user_content=MockContent({"text": "Execute tool"}),
        actions=mock_event_actions,
    )


@pytest.fixture
def mock_base_tool() -> MockBaseTool:
    """Create a mock tool with default name."""
    return MockBaseTool(name="test_tool")


@pytest.fixture
def mock_memory_callback_context() -> MockMemoryCallbackContext:
    """Create a mock callback context that succeeds."""
    return MockMemoryCallbackContext()


@pytest.fixture
def mock_memory_callback_context_no_service() -> MockMemoryCallbackContext:
    """Create a mock callback context that raises ValueError (no service)."""
    return MockMemoryCallbackContext(
        should_raise=ValueError,
        error_message="Cannot add session to memory: memory service is not available.",
    )


@pytest.fixture
def mock_memory_callback_context_with_runtime_error() -> MockMemoryCallbackContext:
    """Create a mock callback context that raises RuntimeError."""
    return MockMemoryCallbackContext(
        should_raise=RuntimeError,
        error_message="Memory service connection failed",
    )


@pytest.fixture
def mock_memory_callback_context_with_attribute_error() -> MockMemoryCallbackContext:
    """Create a mock callback context that raises AttributeError."""
    return MockMemoryCallbackContext(
        should_raise=AttributeError,
        error_message="'MockMemoryCallbackContext' has no invocation context",
    )


@pytest.fixture
def mock_readonly_context() -> MockReadonlyContext:
    """Create a mock readonly context for InstructionProvider testing."""
    return MockReadonlyContext(
        agent_name="instruction_test_agent",
        invocation_id="readonly-inv-123",
        state={"user_tier": "premium", "language": "en"},
    )


# Config testing fixtures
@pytest.fixture
def valid_server_env() -> dict[str, str]:
    """Valid environment variables for ServerEnv model.

    Returns:
        Dictionary with minimal required fields for ServerEnv.
    """
    return {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "AGENT_NAME": "test-agent",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
    }


# Note: clean_environment fixture removed because pytest_configure now sets
# all test environment variables and prevents .env loading. Test isolation is
# achieved by explicit test values set at session start.


@pytest.fixture
def mock_load_dotenv(mocker: MockerFixture) -> MockType:
    """Mock load_dotenv function for testing.

    Returns:
        Mock object for load_dotenv function.
    """
    return mocker.patch("agent_foundation.utils.config.load_dotenv")


@pytest.fixture
def mock_sys_exit(mocker: MockerFixture) -> MockType:
    """Mock sys.exit with SystemExit side effect for testing validation failures.

    Returns:
        Mock object for sys.exit that raises SystemExit(1).
    """
    return mocker.patch("sys.exit", side_effect=SystemExit(1))


@pytest.fixture
def mock_print_config(mocker: MockerFixture) -> Callable[[type], MockType]:
    """Factory fixture for mocking print_config on any model class.

    Returns:
        Function that patches print_config on a given model class.
    """

    def _mock_print_config(model_class: type) -> MockType:
        """Patch print_config on a model class.

        Args:
            model_class: The Pydantic model class to mock print_config on.

        Returns:
            Mock object for the print_config method.
        """
        return mocker.patch.object(model_class, "print_config", autospec=True)

    return _mock_print_config
# Google Chat fixtures
@pytest.fixture
def chat_message_event() -> dict[str, Any]:
    """Sample Google Chat MESSAGE event with realistic content."""
    return {
        "type": "MESSAGE",
        "message": {
            "text": "Show me my timecards for this week",
            "sender": {"displayName": "Test User"},
            "space": {"name": "spaces/TEST123"},
            "thread": {"name": "spaces/TEST123/threads/THREAD1"},
        },
        "user": {"name": "users/TEST_USER", "displayName": "Test User"},
        "space": {"name": "spaces/TEST123", "type": "DM"},
    }


@pytest.fixture
def simple_message_event() -> dict[str, Any]:
    """Simple Google Chat MESSAGE event for error testing.

    Minimal event structure used in tests that focus on error handling
    rather than message content.
    """
    return {
        "type": "MESSAGE",
        "message": {"text": "test"},
        "user": {"name": "users/TEST", "displayName": "Test"},
        "space": {"name": "spaces/TEST", "type": "DM"},
    }


@pytest.fixture
def reset_command_event() -> dict[str, Any]:
    """Google Chat MESSAGE event with /reset command.

    Used to test session reset functionality across unit and integration tests.
    """
    return {
        "type": "MESSAGE",
        "message": {"text": "/reset"},
        "user": {"name": "users/TEST_USER", "displayName": "Test User"},
        "space": {"name": "spaces/TEST", "type": "DM"},
    }


@pytest.fixture
def adk_sse_events() -> list[dict[str, Any]]:
    """Sample SSE event stream from ADK /run_sse endpoint."""
    return [
        {
            "type": "message",
            "content": {
                "parts": [{"text": "Querying Salesforce for your timecards..."}]
            },
        },
        {
            "type": "message",
            "content": {"parts": [{"text": "Found 3 timecards for this week."}]},
        },
        {"type": "end"},
    ]


class MockHttpxStreamResponse:
    """Mock httpx streaming response for SSE testing.

    Mimics the behavior of httpx.Response when used with client.stream().
    Supports async context manager protocol and aiter_lines() for SSE parsing.
    """

    def __init__(self, sse_lines: list[str]) -> None:
        """Initialize mock stream response with SSE lines.

        Args:
            sse_lines: List of SSE-formatted lines to yield from aiter_lines()
        """
        self._sse_lines = sse_lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        """Async iterator yielding SSE lines.

        Mimics httpx.Response.aiter_lines() for Server-Sent Events parsing.

        Yields:
            SSE-formatted lines (e.g., "data: {...}", ": comment", "event: type")
        """
        for line in self._sse_lines:
            yield line

    async def __aenter__(self) -> MockHttpxStreamResponse:
        """Async context manager entry.

        Returns:
            Self (the mock response object)
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit.

        Args:
            *args: Exception info (exc_type, exc_val, exc_tb)
        """
        return None


class MockHttpxClient:
    """Mock httpx.AsyncClient for testing SSE streaming.

    Mimics httpx.AsyncClient behavior with support for:
    - Async context manager protocol
    - stream() method that returns async context manager
    - Exception raising for error cases

    This explicit class-based approach is preferred over manually assigning
    __aenter__/__aexit__ to AsyncMock objects, following the same pattern
    as other test mocks (MockState, MockContent, etc.).
    """

    def __init__(
        self, sse_lines: list[str] | None = None, exception: Exception | None = None
    ) -> None:
        """Initialize mock client with controlled behavior.

        Args:
            sse_lines: SSE lines to return (for success cases)
            exception: Exception to raise when stream() is called (for error cases)
        """
        self._sse_lines = sse_lines or []
        self._exception = exception

    def stream(self, *args: Any, **kwargs: Any) -> MockHttpxStreamResponse:
        """Mock stream method that returns async context manager.

        Args:
            *args: Positional arguments (method, url, etc.) - ignored in mock
            **kwargs: Keyword arguments (json, timeout, etc.) - ignored in mock

        Returns:
            MockHttpxStreamResponse that yields SSE lines

        Raises:
            Exception: The configured exception if set during initialization
        """
        if self._exception:
            raise self._exception
        return MockHttpxStreamResponse(self._sse_lines)

    async def __aenter__(self) -> MockHttpxClient:
        """Async context manager entry.

        Returns:
            Self (the mock client object)
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit.

        Args:
            *args: Exception info (exc_type, exc_val, exc_tb)
        """
        return None


@pytest.fixture
def mock_httpx_client(
    mocker: MockerFixture,
) -> Callable[[list[str] | None, Exception | None], None]:
    """Factory fixture for mocking httpx.AsyncClient with SSE streaming.

    Returns a factory function that creates and patches httpx.AsyncClient with
    a custom mock class that accurately mimics httpx streaming behavior.

    Usage:
        # Success case with SSE lines
        mock_httpx_client(sse_lines=['data: {...}'], exception=None)

        # Error case with exception
        mock_httpx_client(sse_lines=None, exception=httpx.TimeoutException("Timeout"))

    Args:
        sse_lines: Optional list of SSE lines to yield from aiter_lines()
        exception: Optional exception to raise when stream() is called

    Returns:
        Factory function that creates and patches httpx.AsyncClient
    """

    def _create_and_patch(
        sse_lines: list[str] | None = None, exception: Exception | None = None
    ) -> None:
        """Create and patch httpx.AsyncClient with specified behavior.

        Args:
            sse_lines: SSE lines to return (for success cases)
            exception: Exception to raise (for error cases)
        """
        mock_client = MockHttpxClient(sse_lines=sse_lines, exception=exception)
        mocker.patch("agent_foundation.chat.httpx.AsyncClient", return_value=mock_client)

    return _create_and_patch


# Session management fixtures
@pytest.fixture
def mock_session(mocker: MockerFixture) -> Callable[[str], MockType]:
    """Factory fixture for creating mock Session objects.

    Returns:
        Function that creates a mock session with specified ID
    """

    def _create_session(session_id: str) -> MockType:
        """Create mock session with ID.

        Args:
            session_id: Session ID to assign

        Returns:
            Mock session object with id attribute
        """
        session = mocker.Mock()
        session.id = session_id
        return session

    return _create_session


@pytest.fixture
def mock_list_sessions_response(
    mocker: MockerFixture,
) -> Callable[[list[MockType]], MockType]:
    """Factory fixture for creating mock ListSessionsResponse objects.

    Returns:
        Function that creates a mock response with specified sessions
    """

    def _create_response(sessions: list[MockType]) -> MockType:
        """Create mock ListSessionsResponse.

        Args:
            sessions: List of mock session objects

        Returns:
            Mock response object with sessions attribute
        """
        response = mocker.Mock()
        response.sessions = sessions
        return response

    return _create_response


@pytest.fixture
def mock_session_service(mocker: MockerFixture) -> MockType:
    """Mock VertexAiSessionService for session management tests.

    Returns:
        Mock session service with async methods pre-configured as AsyncMock.
    """
    service = mocker.Mock()
    service.list_sessions = mocker.AsyncMock()
    service.create_session = mocker.AsyncMock()
    service.delete_session = mocker.AsyncMock()
    return service  # type: ignore[no-any-return]


@pytest.fixture
def mock_create_session_service(mocker: MockerFixture) -> Callable[..., None]:
    """Mock create_session_service to prevent real API calls in chat tests.

    Patches agent_foundation.chat.create_session_service to return a mock session service
    that returns a test session ID without making API calls.

    Returns:
        Factory function that creates and patches create_session_service
    """

    def _create_and_patch(session_id: str = "test-session-123") -> None:
        """Create and patch create_session_service with specified session ID.

        Args:
            session_id: Session ID to return from get_or_create_session
        """
        # Create mock session object
        mock_session = mocker.Mock()
        mock_session.id = session_id

        # Create mock list result
        mock_list_result = mocker.Mock()
        mock_list_result.sessions = [mock_session]

        # Create mock session service
        mock_service = mocker.Mock()
        mock_service.list_sessions = mocker.AsyncMock(return_value=mock_list_result)
        mock_service.create_session = mocker.AsyncMock(return_value=mock_session)
        mock_service.delete_session = mocker.AsyncMock()

        # Patch create_session_service to return mock service
        mocker.patch("agent_foundation.chat.create_session_service", return_value=mock_service)

    return _create_and_patch


# OpenTelemetry fixtures
def _require_active_span(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that raises if span method called after __exit__.

    Args:
        method: The span method to wrap

    Returns:
        Wrapped method that checks span lifecycle before executing
    """

    @functools.wraps(method)
    def wrapper(self: MockSpan, *args: Any, **kwargs: Any) -> Any:
        if self._exited:
            raise RuntimeError("Span accessed after __exit__")
        return method(self, *args, **kwargs)

    return wrapper


class MockSpan:
    """Mock OpenTelemetry span for testing instrumentation.

    Mimics the behavior of opentelemetry.trace.Span with support for:
    - Context manager protocol (for use with 'with' statements)
    - Attribute setting via set_attribute()
    - Status setting via set_status()
    - Exception recording via record_exception()
    - Strict lifecycle enforcement (raises if accessed after __exit__)

    This class-based approach follows the same pattern as MockHttpxClient
    and other test mocks in this file.

    Strict mode is always enabled to catch bugs where spans are accessed
    outside their context manager scope, even though real OTel spans don't
    raise errors. This enforces proper instrumentation patterns.
    """

    def __init__(self) -> None:
        """Initialize mock span with tracking for method calls."""
        self.attributes: dict[str, Any] = {}
        self.status: Any = None
        self.exceptions: list[Exception] = []
        self._exited = False

    @_require_active_span
    def set_attribute(self, key: str, value: Any) -> None:
        """Record span attribute.

        Args:
            key: Attribute name (e.g., "chat.user.id")
            value: Attribute value

        Raises:
            RuntimeError: If called after __exit__
        """
        self.attributes[key] = value

    @_require_active_span
    def set_status(self, status: Any) -> None:
        """Record span status.

        Args:
            status: Status object (e.g., trace.Status(trace.StatusCode.ERROR, "msg"))

        Raises:
            RuntimeError: If called after __exit__
        """
        self.status = status

    @_require_active_span
    def record_exception(self, exception: Exception) -> None:
        """Record exception in span.

        Args:
            exception: Exception instance to record

        Raises:
            RuntimeError: If called after __exit__
        """
        self.exceptions.append(exception)

    def __enter__(self) -> MockSpan:
        """Context manager entry.

        Returns:
            Self (the mock span object)
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit.

        Args:
            *args: Exception info (exc_type, exc_val, exc_tb)
        """
        self._exited = True
        return None


class MockTracer:
    """Mock OpenTelemetry tracer for testing instrumentation.

    Mimics the behavior of opentelemetry.trace.Tracer with support for:
    - start_as_current_span() that returns context manager
    - Tracking of created spans for test assertions
    - Strict lifecycle enforcement (all spans raise after __exit__)

    This class-based approach is preferred over manually creating nested
    mocks, following the same pattern as MockHttpxClient and other mocks.

    All created spans enforce proper context manager usage by raising
    RuntimeError if accessed after __exit__. This catches bugs even though
    real OTel spans don't raise errors.
    """

    def __init__(self) -> None:
        """Initialize mock tracer with span tracking."""
        self.spans: list[tuple[str, MockSpan]] = []

    def start_as_current_span(self, name: str, **kwargs: Any) -> MockSpan:
        """Create and track a new span.

        Args:
            name: Span name (e.g., "handle_chat_message")
            **kwargs: Optional keyword arguments (e.g., attributes={"key": "value"})

        Returns:
            MockSpan that can be used as context manager
        """
        span = MockSpan()
        if "attributes" in kwargs:
            for key, value in kwargs["attributes"].items():
                span.set_attribute(key, value)
        self.spans.append((name, span))
        return span


@pytest.fixture
def mock_tracer(mocker: MockerFixture) -> MockTracer:
    """Mock OpenTelemetry tracer for testing instrumentation.

    Creates MockTracer instance that enforces proper context manager usage.
    All spans raise RuntimeError if accessed after __exit__, catching bugs
    where spans are used outside their scope (even though real OTel spans
    don't raise errors).

    Patches into both chat.py and session.py modules to intercept span
    creation and attribute setting.

    Usage in tests:
        def test_something(mock_tracer):
            # Call function that creates spans
            result = await handle_chat_message(...)

            # Verify span creation
            assert len(mock_tracer.spans) == 3
            span_names = [name for name, _ in mock_tracer.spans]
            assert "handle_chat_message" in span_names

            # Verify attributes on specific span
            for name, span in mock_tracer.spans:
                if name == "handle_chat_message":
                    assert span.attributes["chat.event.type"] == "MESSAGE"
                    assert span.attributes["agent.name"] == "agent_foundation"

            # Verify exception recording
            for name, span in mock_tracer.spans:
                if name == "handle_chat_message":
                    assert len(span.exceptions) > 0

    Returns:
        MockTracer instance that enforces proper span lifecycle
    """
    mock = MockTracer()
    mocker.patch("agent_foundation.chat.tracer", mock)
    mocker.patch("agent_foundation.session.tracer", mock)
    return mock
