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
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool

load_dotenv()

server = Server()
memories = {}

def get_memory(context: Context) -> UnconstrainedMemory:
    """Get or create session memory"""
    session_id = getattr(context, "session_id", "default")
    return memories.setdefault(session_id, UnconstrainedMemory())

def extract_citations(text: str, search_results=None) -> tuple[list[CitationMetadata], str]:
    """Extract citations and clean text"""
    citations, offset = [], 0
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    
    for match in re.finditer(pattern, text):
        content, url = match.groups()
        start = match.start() - offset
        
        citations.append(CitationMetadata(
            kind="citation", url=url,
            title=url.split("/")[-1].replace("-", " ").title() or content[:50],
            description=content[:100] + ("..." if len(content) > 100 else ""),
            start_index=start, end_index=start + len(content)
        ))
        offset += len(match.group(0)) - len(content)
    
    return citations, re.sub(pattern, r"\1", text)

def is_casual(msg: str) -> bool:
    """Check if message is casual/greeting"""
    casual_words = {'hey', 'hi', 'hello', 'thanks', 'bye', 'cool', 'nice', 'ok', 'yes', 'no'}
    words = msg.lower().strip().split()
    return len(words) <= 3 and any(w in casual_words for w in words)

@server.agent(
    name="jennas_granite_chat",
    description="Friendly Granite-powered chat assistant with search, citations, and trajectory tracking.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi! I'm your Granite-powered AI assistant—here to help with questions, research, and more. What can I do for you today?",
                display_name="Jenna's Granite Chat",
                tools=[
                    AgentToolInfo(name="Think", description="Reasoning and analysis for complex topics"),
                    AgentToolInfo(name="DuckDuckGo", description="Web search for current information")
                ]
            )
        ),
        author={"name": "Jenna Winkler"},
        contributors=[{"name": "Tomas Weiss"}, {"name": "Tomas Dvorak"}],
        recommended_models=["granite3.3:8b-beeai"],
        tags=["Granite", "Chat", "Research"], framework="BeeAI", license="Apache 2.0"
    )
)
async def general_chat_assistant(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """General chat assistant with search and citations"""
    
    user_msg = input[-1].parts[0].content if input else "Hello"
    memory = get_memory(context)
    
    yield MessagePart(metadata=TrajectoryMetadata(
        kind="trajectory", key=str(uuid.uuid4()), 
        message=f"💬 Processing: '{user_msg}'"
    ))
    
    try:
        await memory.add(UserMessage(user_msg))
        
        # Setup LLM
        OpenAIChatModel.tool_choice_support = set()
        llm = OpenAIChatModel(
            model_id=os.getenv('LLM_MODEL', 'llama3.1'),
            base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "dummy")
        )
        
        # Setup agent with conditional requirements
        agent = RequirementAgent(
            llm=llm, memory=memory,
            tools=[ThinkTool(), DuckDuckGoSearchTool()],
            requirements=[
                ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False),
                ConditionalRequirement(
                    DuckDuckGoSearchTool, max_invocations=2, consecutive_allowed=False,
                    custom_checks=[lambda state: not is_casual(user_msg)]
                )
            ],
            instructions="""You are a helpful AI assistant. For search results, ALWAYS use proper markdown citations: [description](URL).

Examples:
- [OpenAI releases GPT-5](https://example.com/gpt5)
- [AI adoption increases 67%](https://example.com/ai-study)

Use DuckDuckGo for current info, facts, and specific questions. Respond naturally to casual greetings without search."""
        )
        
        yield MessagePart(metadata=TrajectoryMetadata(
            kind="trajectory", key=str(uuid.uuid4()),
            message="🛠️ Granite Chat ready with Think and Search tools"
        ))
        
        response_text = ""
        search_results = None
        
        # Run agent
        async for event, meta in agent.run(
            user_msg,
            execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2, total_max_retries=5),
            expected_output="Markdown format with proper [text](URL) citations for search results."
        ):
            if meta.name == "success" and event.state.steps:
                step = event.state.steps[-1]
                if not step.tool:
                    continue
                    
                tool_name = step.tool.name
                
                if tool_name == "final_answer":
                    response_text += step.input["response"]
                elif tool_name == "DuckDuckGo":
                    search_results = getattr(step.output, 'results', None)
                    query = step.input.get("query", "Unknown")
                    count = len(search_results) if search_results else 0
                    
                    yield MessagePart(metadata=TrajectoryMetadata(
                        tool_name="DuckDuckGo",
                        tool_input={"query": query, "results_count": count},
                        message=f"🔍 Searched: '{query}' → {count} results"
                    ))
                elif tool_name == "think":
                    yield MessagePart(metadata=TrajectoryMetadata(
                        message=step.input["thoughts"], tool_name="Thought"
                    ))
        
        await memory.add(AssistantMessage(response_text))
        
        # Extract citations and yield response
        citations, clean_text = extract_citations(response_text, search_results)
        
        yield MessagePart(content=clean_text)
        for citation in citations:
            yield MessagePart(metadata=citation)
            
        yield MessagePart(metadata=TrajectoryMetadata(
            kind="trajectory", key=str(uuid.uuid4()),
            message="✅ Response completed"
        ))
        
    except Exception as e:
        print(f"❌ Error: {e}\n{traceback.format_exc()}")
        yield MessagePart(metadata=TrajectoryMetadata(
            kind="trajectory", key=str(uuid.uuid4()), message=f"❌ Error: {e}"
        ))
        yield MessagePart(content=f"🚨 Error processing request: {e}")

def run():
    """Start the server"""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()