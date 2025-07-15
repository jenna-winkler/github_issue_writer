import os
import re
import uuid
from collections.abc import AsyncGenerator

from dotenv import load_dotenv

from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.models import CitationMetadata, TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
model = os.getenv("LLM_MODEL", "ollama:granite3.3:8b-beeai")

server = Server()

conversation_memories = {}


class TrajectoryCapture:
    """Captures trajectory steps for display"""

    def __init__(self):
        self.steps = []

    def write(self, message: str) -> int:
        self.steps.append(message.strip())
        return len(message)


class TrackedTool:
    """Base class for tool tracking"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.results = []

    def add_result(self, result):
        self.results.append(result)


class TrackedDuckDuckGoTool(DuckDuckGoSearchTool):
    """DuckDuckGo tool with result tracking"""

    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker

    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(("DuckDuckGo", result))
        return result


class TrackedWikipediaTool(WikipediaTool):
    """Wikipedia tool with result tracking"""

    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker

    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(("Wikipedia", result))
        return result


class TrackedOpenMeteoTool(OpenMeteoTool):
    """Weather tool with result tracking"""

    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker

    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(("OpenMeteo", result))
        return result


def get_session_id(context: Context) -> str:
    """Extract session ID from context, fallback to default if not available"""
    session_id = getattr(context, "session_id", None)
    if not session_id:
        session_id = getattr(context, "headers", {}).get("session-id", "default")
    return str(session_id)


def get_or_create_memory(session_id: str) -> UnconstrainedMemory:
    """Get existing memory for session or create new one"""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = UnconstrainedMemory()
    return conversation_memories[session_id]


def extract_citations_from_response(response_text: str, tool_tracker: TrackedTool = None) -> tuple[list[CitationMetadata], str]:
    """Extract citations from response text and return CitationMetadata objects with cleaned text"""
    citations = []
    cleaned_text = response_text
    offset = 0

    # Handle explicit markdown-style citations first
    citation_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    for match in re.finditer(citation_pattern, response_text):
        content = match.group(1)
        url = match.group(2)
        original_start = match.start()

        adjusted_start = original_start - offset
        adjusted_end = adjusted_start + len(content)

        title = url.split("/")[-1].replace("-", " ").title()
        if not title or title == "":
            title = content[:50] + "..." if len(content) > 50 else content

        citation = CitationMetadata(
            kind="citation",
            url=url,
            title=title,
            description=content[:100] + "..." if len(content) > 100 else content,
            start_index=adjusted_start,
            end_index=adjusted_end,
        )
        citations.append(citation)

        removed_chars = len(match.group(0)) - len(content)
        offset += removed_chars

    cleaned_text = re.sub(citation_pattern, r"\1", response_text)

    # Add automatic citations from tool results
    if tool_tracker:
        for tool_name, tool_result in tool_tracker.results:
            if tool_name == "DuckDuckGo" and hasattr(tool_result, 'search_results'):
                for search_result in tool_result.search_results:
                    if hasattr(search_result, 'url') and hasattr(search_result, 'title'):
                        # Create a citation for this DuckDuckGo result
                        citation = CitationMetadata(
                            kind="citation",
                            url=search_result.url,
                            title=search_result.title,
                            description=getattr(search_result, 'description', '')[:100] + "..." if len(getattr(search_result, 'description', '')) > 100 else getattr(search_result, 'description', ''),
                            start_index=None,  # Will be positioned at the end
                            end_index=None,
                        )
                        citations.append(citation)
                        
            elif tool_name == "Wikipedia" and hasattr(tool_result, 'search_results'):
                for search_result in tool_result.search_results:
                    if hasattr(search_result, 'url') and hasattr(search_result, 'title'):
                        citation = CitationMetadata(
                            kind="citation",
                            url=search_result.url,
                            title=search_result.title,
                            description=getattr(search_result, 'description', '')[:100] + "..." if len(getattr(search_result, 'description', '')) > 100 else getattr(search_result, 'description', ''),
                            start_index=None,
                            end_index=None,
                        )
                        citations.append(citation)
                        
            elif tool_name == "OpenMeteo":
                # Add a generic weather citation
                citation = CitationMetadata(
                    kind="citation",
                    url="https://open-meteo.com/",
                    title="Open-Meteo Weather API",
                    description="Weather forecast data",
                    start_index=None,
                    end_index=None,
                )
                citations.append(citation)

    return citations, cleaned_text


@server.agent(
    name="jennas_granite_chat",
    description="A friendly and knowledgeable general chat assistant powered by Granite. Features the RequirementAgent from the BeeAI Framework with dynamic citations and trajectory tracking. Built with conditional tool requirements, session memory management, and real-time citation extraction from Wikipedia, DuckDuckGo, and OpenMeteo sources. Provides transparent reasoning through trajectory visualization and maintains conversation context across sessions.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi! I'm your Granite-powered AI assistant—here to help with questions, research, weather, and more. What can I do for you today?",
                display_name="Jenna's Granite Chat",
                tools=[
                    AgentToolInfo(
                        name="Think",
                        description="Advanced reasoning and analysis to provide thoughtful, well-structured responses to complex questions and topics.",
                    ),
                    AgentToolInfo(
                        name="Wikipedia",
                        description="Search comprehensive information from Wikipedia's vast knowledge base for factual information, definitions, and explanations.",
                    ),
                    AgentToolInfo(
                        name="Weather",
                        description="Get current weather conditions, forecasts, and climate information for any location worldwide.",
                    ),
                    AgentToolInfo(
                        name="DuckDuckGo",
                        description="Search the web for current information, news, and real-time updates on any topic.",
                    ),
                ],
            )
        ),
        author={"name": "Jenna Winkler"},
        contributors=[{"name": "Tomas Weiss"}],
        recommended_models=["granite3.3:8b-beeai"],
        tags=["Granite", "Chat", "Research"],
        framework="BeeAI",
        programming_language="Python",
        license="Apache 2.0",
    ),
)
async def general_chat_assistant(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    General chat assistant that provides:
    - Friendly, knowledgeable conversation
    - Dynamic citations from reliable sources
    - Trajectory tracking for transparency
    - Multi-tool integration for comprehensive responses
    """

    user_message = input[-1].parts[0].content if input else "Hello"
    session_id = get_session_id(context)

    tool_tracker = TrackedTool("general_chat")
    trajectory = TrajectoryCapture()

    session_memory = get_or_create_memory(session_id)

    yield MessagePart(
        metadata=TrajectoryMetadata(
            kind="trajectory", key=str(uuid.uuid4()), message=f"💬 Processing your message: '{user_message}'"
        )
    )

    try:
        await session_memory.add(UserMessage(user_message))

        tracked_duckduckgo = TrackedDuckDuckGoTool(tool_tracker)
        tracked_wikipedia = TrackedWikipediaTool(tool_tracker)
        tracked_weather = TrackedOpenMeteoTool(tool_tracker)

        agent = RequirementAgent(
            llm=ChatModel.from_name(model),
            memory=session_memory,
            tools=[ThinkTool(), tracked_wikipedia, tracked_weather, tracked_duckduckgo],
            requirements=[
                ConditionalRequirement(
                    ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False, max_invocations=3
                ),
                ConditionalRequirement(tracked_wikipedia, max_invocations=1, consecutive_allowed=False),
                ConditionalRequirement(tracked_weather, max_invocations=1, consecutive_allowed=False),
                ConditionalRequirement(tracked_duckduckgo, max_invocations=1, consecutive_allowed=False),
            ],
            instructions="""
            You are Granite Chat, a friendly and knowledgeable AI assistant powered by Granite.

            Your personality:
            - Warm, approachable, and conversational
            - Curious and enthusiastic about helping
            - Clear and informative in explanations
            - Adaptive to the user's tone and needs

            Your capabilities:
            - Engage in natural, flowing conversations
            - Provide accurate, well-researched information
            - Help with questions across many topics
            - Offer explanations, advice, and insights

            How to respond:
            1. First, think about what the user is asking and what information would be most helpful
            2. Use tools when needed to provide accurate, current information:
               - Wikipedia for factual information, definitions, and comprehensive knowledge
               - DuckDuckGo for current events, recent information, and web searches
               - OpenMeteo for weather-related questions
            3. When you reference information from tool results, the system will automatically add citations
            4. Provide conversational, engaging responses that directly address the user's needs

            Response guidelines:
            - Be conversational and natural, not robotic
            - Reference specific facts and information from your tool results when relevant
            - Acknowledge when you're uncertain or when information might be incomplete
            - Ask follow-up questions when it would be helpful
            - Adapt your level of detail to match the user's apparent needs

            !!!IMPORTANT CITATION RULES!!!
            - When you use information from Wikipedia, DuckDuckGo, or OpenMeteo, try to cite it using this format: [Information](URL)
            - If you reference specific facts from your tool results, format them as citations when possible
            - Examples:
              - [The first SMS was sent in 1992](https://example.com/url) according to search results
              - [Quantum computing](https://en.wikipedia.org/wiki/Quantum_computing) is a rapidly advancing field
              - The weather data shows [15°C with light rain](https://open-meteo.com/)
            - If you don't have specific URLs, don't worry - the system will automatically add citations for tool results
            - Focus on being helpful and informative first, citations second
            """,
        )

        yield MessagePart(
            metadata=TrajectoryMetadata(
                kind="trajectory",
                key=str(uuid.uuid4()),
                message="🛠️ Granite Chat initialized with Think, Wikipedia, Weather, and Search tools",
            )
        )

        response = await agent.run(
            user_message, execution=AgentExecutionConfig(max_iterations=10, max_retries_per_step=2, total_max_retries=5)
        ).middleware(GlobalTrajectoryMiddleware(target=trajectory, included=[Tool]))

        response_text = response.answer.text

        await session_memory.add(AssistantMessage(response_text))

        for i, step in enumerate(trajectory.steps):
            if step.strip():
                tool_name = None
                if "ThinkTool" in step:
                    tool_name = "Think"
                elif "WikipediaTool" in step:
                    tool_name = "Wikipedia"
                elif "OpenMeteoTool" in step:
                    tool_name = "Weather"
                elif "DuckDuckGo" in step:
                    tool_name = "DuckDuckGo"

                yield MessagePart(
                    metadata=TrajectoryMetadata(
                        kind="trajectory", key=str(uuid.uuid4()), message=f"Step {i + 1}: {step}", tool_name=tool_name
                    )
                )

        citations, cleaned_response_text = extract_citations_from_response(response_text, tool_tracker)

        yield MessagePart(content=cleaned_response_text)

        for citation in citations:
            yield MessagePart(metadata=citation)

        yield MessagePart(
            metadata=TrajectoryMetadata(
                kind="trajectory",
                key=str(uuid.uuid4()),
                message="✅ Response completed successfully",
            )
        )

    except Exception as e:
        yield MessagePart(
            metadata=TrajectoryMetadata(kind="trajectory", key=str(uuid.uuid4()), message=f"❌ Error: {str(e)}")
        )
        yield MessagePart(content=f"🚨 I apologize, but I encountered an error while processing your request: {str(e)}")


def run():
    """Entry point for the server."""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()