import os
import re
import traceback
from typing import Annotated
from textwrap import dedent

from beeai_framework.adapters.openai import OpenAIChatModel
from dotenv import load_dotenv

from a2a.types import AgentCapabilities, AgentSkill, Message
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.a2a.extensions import AgentDetail, AgentDetailTool, CitationExtensionServer, CitationExtensionSpec, TrajectoryExtensionServer, TrajectoryExtensionSpec, LLMServiceExtensionServer, LLMServiceExtensionSpec
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_sdk.util.file import load_file

load_dotenv()

server = Server()
memories = {}

def get_memory(context: RunContext) -> UnconstrainedMemory:
    """Get or create session memory"""
    
    context_id = getattr(context, "context_id", getattr(context, "session_id", "default"))
    return memories.setdefault(context_id, UnconstrainedMemory())

def extract_citations(text: str, search_results=None) -> tuple[list[dict], str]:
    """Extract citations and clean text - returns citations in the correct format"""
    citations, offset = [], 0
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    
    for match in re.finditer(pattern, text):
        content, url = match.groups()
        start = match.start() - offset

        citations.append({
            "url": url,
            "title": url.split("/")[-1].replace("-", " ").title() or content[:50],
            "description": content[:100] + ("..." if len(content) > 100 else ""),
            "start_index": start, 
            "end_index": start + len(content)
        })
        offset += len(match.group(0)) - len(content)

    return citations, re.sub(pattern, r"\1", text)

def is_casual(msg: str) -> bool:
    """Check if message is casual/greeting"""
    casual_words = {'hey', 'hi', 'hello', 'thanks', 'bye', 'cool', 'nice', 'ok', 'yes', 'no'}
    words = msg.lower().strip().split()
    return len(words) <= 3 and any(w in casual_words for w in words)

@server.agent(
    name="Jenna's Granite Chat",
    default_input_modes=["text", "text/plain", "application/pdf", "text/csv", "application/json"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi! I'm your Granite-powered AI assistant. How can I help?",
        version="0.0.10",
        tools=[
            AgentDetailTool(
                name="Think", 
                description="Advanced reasoning and analysis to provide thoughtful, well-structured responses to complex questions and topics."
            ),
            AgentDetailTool(
                name="DuckDuckGo", 
                description="Search the web for current information, news, and real-time updates on any topic."
            ),
            AgentDetailTool(
                name="File Processing", 
                description="Read and analyze uploaded files including PDFs, text files, CSV data, and JSON documents."
            )
        ],
        framework="BeeAI",
        author={
            "name": "Jenna Winkler"
        },
        source_code_url="https://github.com/jenna-winkler/granite_chat"
    ),
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="chat",
            name="Chat",
            description=dedent(
                """\
                The agent is an AI-powered conversational system designed to process user messages, maintain context,
                generate intelligent responses, and analyze uploaded files.
                """
            ),
            tags=["Chat", "Files"],
            examples=[
                "What are the latest advancements in AI research from 2025?",
                "What's the difference between LLM tool use and API orchestration?",
                "Can you help me draft an email apologizing for missing a meeting?",
                "Analyze this CSV file and tell me the key trends.",
                "Summarize the main points from this PDF document.",
            ]

        )
    ],
)
async def general_chat_assistant(
    input: Message, 
    context: RunContext,
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer, 
        LLMServiceExtensionSpec.single_demand(
            suggested=("ibm/granite-3-3-8b-instruct", "llama3.1", "gpt-4o-mini")
        )
    ]
):
    """
    This is a general-purpose chat assistant prototype built with the BeeAI Framework and powered by Granite. It demonstrates advanced capabilities of both the BeeAI Framework and BeeAI SDK.

    ### BeeAI Framework Features

    - **RequirementAgent:** An experimental agent that selects and executes tools based on defined rules instead of relying solely on LLM decisions. ConditionalRequirement rules determine when and how each tool is used.
    - **ThinkTool:** Provides advanced reasoning and structured analysis.
    - **DuckDuckGoSearchTool:** Performs real-time web searches with invocation limits and casual message detection.
    - **Memory Management:** Uses `UnconstrainedMemory` to maintain full conversation context with session persistence.
    - **Error Handling:** Try-catch blocks provide clear messages; `is_casual()` skips unnecessary tool calls for simple messages.

    ### BeeAI SDK Features

    - **GUI Configuration:** Configures agent details including interaction mode, user greeting, tool descriptions, and metadata through AgentDetail.
    - **TrajectoryMetadata:** Logs agent decisions and tool execution for transparency.
    - **CitationMetadata:** Converts markdown links into structured objects for GUI display.
    - **File Processing:** Supports text, PDF, CSV, and JSON files.
    - **LLM Service Extension:** Uses platform-managed LLMs for consistent model access.
    """

    user_msg = ""
    file_content = ""
    uploaded_files = []
    
    for part in input.parts:
        part_root = part.root
        if part_root.kind == "text":
            user_msg = part_root.text
        elif part_root.kind == "file":
            uploaded_files.append(part_root)
    
    if not user_msg:
        user_msg = "Hello"
    
    memory = get_memory(context)
    
    if uploaded_files:
        yield trajectory.trajectory_metadata(
            title="Processing Files",
            content=f"📁 Processing {len(uploaded_files)} uploaded file(s)"
        )
        
        for file_part in uploaded_files:
            try:
                async with load_file(file_part) as loaded_content:
                    filename = file_part.file.name
                    content_type = file_part.file.mime_type
                    
                    content = loaded_content.text
                    file_content += f"\n\n--- File: {filename} ({content_type}) ---\n{content}\n"
                    
                    yield trajectory.trajectory_metadata(
                        title="File Loaded",
                        content=f"📄 Loaded: {filename} ({len(content)} characters)"
                    )
                    
            except Exception as e:
                yield trajectory.trajectory_metadata(
                    title="File Error",
                    content=f"❌ Error loading {file_part.file.name}: {e}"
                )
    
    full_message = user_msg
    if file_content:
        full_message += f"\n\nUploaded file content:{file_content}"
    
    yield trajectory.trajectory_metadata(
        title="Processing Message",
        content=f"💬 Processing: '{user_msg}'" + (f" with {len(uploaded_files)} file(s)" if uploaded_files else "")
    )
    
    try:
        await memory.add(UserMessage(full_message))
        
        if llm:
            llm_config = llm.data.llm_fulfillments.get("default")
            
            if llm_config:
                OpenAIChatModel.tool_choice_support = set()
                llm_client = OpenAIChatModel(
                    model_id=llm_config.api_model,
                    base_url=llm_config.api_base,
                    api_key=llm_config.api_key
                )
            else:                
                OpenAIChatModel.tool_choice_support = set()
                llm_client = OpenAIChatModel(
                    model_id=os.getenv('LLM_MODEL', 'llama3.1'),
                    base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
                    api_key=os.getenv("LLM_API_KEY", "dummy")
                )
        else:
            yield trajectory.trajectory_metadata(
                title="LLM Configuration",
                content="⚠️ LLM Service Extension not available, using environment config"
            )
            
            OpenAIChatModel.tool_choice_support = set()
            llm_client = OpenAIChatModel(
                model_id=os.getenv('LLM_MODEL', 'llama3.1'),
                base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
                api_key=os.getenv("LLM_API_KEY", "dummy")
            )
        
        agent = RequirementAgent(
            llm=llm_client, 
            memory=memory,
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

Use DuckDuckGo for current info, facts, and specific questions. Respond naturally to casual greetings without search.

When files are uploaded, analyze and summarize their content. For data files (CSV/JSON), highlight key insights and patterns."""
        )
        
        yield trajectory.trajectory_metadata(
            title="Agent Ready",
            content="🛠️ Granite Chat ready with Think, Search tools" + (" and file processing" if uploaded_files else "")
        )
        
        response_text = ""
        search_results = None
        
        async for event, meta in agent.run(
            full_message,
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
                    
                    yield trajectory.trajectory_metadata(
                        title="DuckDuckGo Search",
                        content=f"🔍 Searched: '{query}' → {count} results"
                    )
                elif tool_name == "think":
                    yield trajectory.trajectory_metadata(
                        title="Thought",
                        content=step.input["thoughts"]
                    )
        
        await memory.add(AssistantMessage(response_text))
        
        citations, clean_text = extract_citations(response_text, search_results)

        yield clean_text
        
        if citations:
            citation_objects = []
            for cit in citations:
                citation_objects.append({
                    "url": cit["url"],
                    "title": cit["title"],
                    "description": cit["description"],
                    "start_index": cit["start_index"],
                    "end_index": cit["end_index"]
                })
            
            yield citation.citation_metadata(citations=citation_objects)
            
        yield trajectory.trajectory_metadata(
            title="Completion",
            content="✅ Response completed"
        )

    except Exception as e:
        print(f"❌ Error: {e}\n{traceback.format_exc()}")
        yield trajectory.trajectory_metadata(
            title="Error",
            content=f"❌ Error: {e}"
        )
        yield f"🚨 Error processing request: {e}"

def run():
    """Start the server"""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()