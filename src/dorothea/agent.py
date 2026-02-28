"""ADK LlmAgent configuration."""

import os
from functools import cached_property

from google import genai
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.google_llm import Gemini
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools import google_search
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from .callbacks import LoggingCallbacks, add_session_to_memory
from .prompt import (
    return_description_root,
    return_global_instruction,
    return_instruction_root,
)

logging_callbacks = LoggingCallbacks()


class GlobalVertexGemini(Gemini):
    @cached_property
    def api_client(self) -> genai.Client:
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="global",
        )

    @cached_property
    def _live_api_client(self) -> genai.Client:
        from google.genai import types

        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="global",
            http_options=types.HttpOptions(
                headers=self._tracking_headers(),
                api_version=self._live_api_version,
            ),
        )


root_agent = LlmAgent(
    name="dorothea",
    description=return_description_root(),
    before_agent_callback=logging_callbacks.before_agent,
    after_agent_callback=[logging_callbacks.after_agent, add_session_to_memory],
    model=GlobalVertexGemini(
        model=os.getenv("ROOT_AGENT_MODEL", "gemini-3-flash-preview")
    ),
    instruction=return_instruction_root(),
    tools=[PreloadMemoryTool(), google_search],
    before_model_callback=logging_callbacks.before_model,
    after_model_callback=logging_callbacks.after_model,
    before_tool_callback=logging_callbacks.before_tool,
    after_tool_callback=logging_callbacks.after_tool,
)

# Optional App configs explicitly set to None for template documentation
app = App(
    name="dorothea",
    root_agent=root_agent,
    plugins=[
        GlobalInstructionPlugin(return_global_instruction),
        LoggingPlugin(),
    ],
    events_compaction_config=None,
    context_cache_config=None,
    resumability_config=None,
)
