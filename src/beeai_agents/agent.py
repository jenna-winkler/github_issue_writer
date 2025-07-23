import os
import re
import uuid
import traceback
from collections.abc import AsyncGenerator

from beeai_framework.adapters.openai import OpenAIChatModel
from dotenv import load_dotenv

from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.models import CitationMetadata, TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool

load_dotenv()

server = Server()
conversation_memories = {}


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


def extract_citations_from_response(response_text: str) -> tuple[list[CitationMetadata], str]:
    """Extract citations from response text and return CitationMetadata objects with cleaned text"""
    citations = []
    cleaned_text = response_text
    offset = 0

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

    return citations, cleaned_text


@server.agent(
    name="jennas_granite_chat",
    description="A friendly and knowledgeable general chat assistant powered by Granite. Features the RequirementAgent from the BeeAI Framework with dynamic citations and trajectory tracking. Built with conditional tool requirements, session memory management, and real-time citation extraction from DuckDuckGo sources. Provides transparent reasoning through trajectory visualization and maintains conversation context across sessions.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi! I'm your Granite-powered AI assistant—here to help with questions, research, and more. What can I do for you today?",
                display_name="Jenna's Granite Chat",
                tools=[
                    AgentToolInfo(
                        name="Think",
                        description="Advanced reasoning and analysis to provide thoughtful, well-structured responses to complex questions and topics.",
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
        tags=["Granite", "Chat", "Research", "Assist"],
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

    session_memory = get_or_create_memory(session_id)

    yield MessagePart(
        metadata=TrajectoryMetadata(
            kind="trajectory", key=str(uuid.uuid4()), message=f"💬 Processing your message: '{user_message}'"
        )
    )

    try:
        await session_memory.add(UserMessage(user_message))

        OpenAIChatModel.tool_choice_support = set()
        llm = OpenAIChatModel(
            model_id=os.getenv('LLM_MODEL', 'llama3.1'),
            base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "dummy"),
        )

        def is_simple_greeting_or_casual(state) -> bool:
            """Check if the user message is a simple greeting or casual conversation that doesn't need search"""
            user_input = user_message.lower().strip()
            simple_patterns = [
                'hey', 'hi', 'hello', 'howdy', 'sup', 'what\'s up', 'whats up',
                'good morning', 'good afternoon', 'good evening', 'good night',
                'how are you', 'how\'s it going', 'hows it going', 'thanks', 'thank you',
                'bye', 'goodbye', 'see you', 'later', 'nice', 'cool', 'awesome',
                'ok', 'okay', 'sure', 'yes', 'no', 'maybe', 'lol', 'haha'
            ]
            
            # Check for exact matches or very short casual responses
            if user_input in simple_patterns or len(user_input.split()) <= 3:
                for pattern in simple_patterns:
                    if pattern in user_input:
                        return True
            return False

        agent = RequirementAgent(
            llm=llm,
            memory=session_memory,
            tools=[ThinkTool(), DuckDuckGoSearchTool()],
            requirements=[
                ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False),
                ConditionalRequirement(
                    DuckDuckGoSearchTool, 
                    max_invocations=1, 
                    consecutive_allowed=False,
                    custom_checks=[lambda state: not is_simple_greeting_or_casual(state)]
                )
            ],
            instructions="""
            You are a knowledgeable and helpful general-purpose assistant designed to provide engaging conversation and answer questions with real-world information when needed.

            Your goal is to provide natural, helpful responses that match the user's intent and conversation style.

            ## Approach:

            1. **Understand the User's Intent**: 
               - For simple greetings, casual conversation, or general questions, respond naturally and conversationally
               - For information requests requiring current data, use the DuckDuckGo search tool
               
            2. **When to Use Tools**:
               - **Think Tool**: Always use this first to understand the user's request and plan your response
               - **DuckDuckGo**: Only use when the user is asking for specific information, current events, facts, or data that requires web search
               
            ## Simple Conversations (No Search Needed):
            - Greetings: "hi", "hey", "hello", "how are you"
            - Casual responses: "thanks", "cool", "nice", "okay"
            - General conversation and small talk
            - Basic questions you can answer conversationally

            ## When to Search:
            - Specific factual questions requiring current information
            - Requests for news, current events, or recent data
            - Product comparisons, restaurant recommendations, travel info
            - Technical information or detailed explanations

            ## Response Format:

            - Be natural, friendly, and conversational in tone
            - For simple conversations, respond directly without unnecessary citations
            - When you do use search results, include proper citations with format: `[Descriptive text](URL)`
            - Match the user's energy and conversation style

            Always think first about what the user really needs - sometimes a friendly, direct response is better than a researched answer.
            """,
        )

        yield MessagePart(
            metadata=TrajectoryMetadata(
                kind="trajectory",
                key=str(uuid.uuid4()),
                message="🛠️ Granite Chat initialized with Think and Search tools",
            )
        )

        response_text = ""

        async for event, meta in agent.run(
            user_message, execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2, total_max_retries=5), expected_output="The answer must be in Markdown format and include citations."
        ):
            if meta.name == "success":
                last_step = event.state.steps[-1] if event.state.steps else None

                if last_step and last_step.tool is not None:
                    if last_step.tool.name == "final_answer":
                        response_text += last_step.input["response"]
                    elif last_step.tool.name == "DuckDuckGo":
                        yield MessagePart(metadata=TrajectoryMetadata(tool_name="DuckDuckGo", tool_input={"query": last_step.input["query"]}))
                    elif last_step.tool.name == "think":
                        yield MessagePart(metadata=TrajectoryMetadata(message=last_step.input["thoughts"], tool_name="Thought"))


        await session_memory.add(AssistantMessage(response_text))

        citations, cleaned_response_text = extract_citations_from_response(response_text)

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
        # Log the full exception with stack trace to console
        print(f"❌ Error in general_chat_assistant: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")

        yield MessagePart(
            metadata=TrajectoryMetadata(kind="trajectory", key=str(uuid.uuid4()), message=f"❌ Error: {str(e)}")
        )
        yield MessagePart(content=f"🚨 I apologize, but I encountered an error while processing your request: {str(e)}")


def run():
    """Entry point for the server."""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()