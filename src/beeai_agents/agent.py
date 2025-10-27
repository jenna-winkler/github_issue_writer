import os
from typing import Annotated

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend.types import ChatModelParameters
from dotenv import load_dotenv

import a2a.types
from a2a.types import AgentSkill, Message
from beeai_sdk.server import Server
from beeai_sdk.server.context import RunContext
from beeai_sdk.a2a.extensions import AgentDetailExtensionSpec, AgentDetail, CitationExtensionServer, CitationExtensionSpec, TrajectoryExtensionServer, TrajectoryExtensionSpec, LLMServiceExtensionServer, LLMServiceExtensionSpec
from beeai_sdk.a2a.extensions.ui.form import TextField, MultiSelectField, OptionItem, FormExtensionServer, FormExtensionSpec, FormRender
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.think import ThinkTool

load_dotenv()

server = Server()

form_render = FormRender(
    id="github_issue_form",
    title="Create a GitHub Issue",
    description="Fill out the details below to generate a well-structured GitHub issue",
    columns=1,
    submit_label="Generate Issue",
    fields=[
        TextField(
            type="text",
            id="title",
            label="Issue Title",
            placeholder="Brief, descriptive title for the issue",
            required=True,
            col_span=1
        ),
        MultiSelectField(
            type="multiselect",
            id="issue_type",
            label="Issue Type",
            required=True,
            col_span=1,
            options=[
                OptionItem(id="bug", label="Bug Report"),
                OptionItem(id="feature", label="Feature Request"),
                OptionItem(id="enhancement", label="Enhancement"),
                OptionItem(id="documentation", label="Documentation")
            ]
        ),
        TextField(
            type="text",
            id="description",
            label="Description",
            placeholder="What is the issue? What needs to be done?",
            required=True,
            col_span=1
        ),
        MultiSelectField(
            type="multiselect",
            id="priority",
            label="Priority",
            required=False,
            col_span=1,
            options=[
                OptionItem(id="low", label="Low"),
                OptionItem(id="medium", label="Medium"),
                OptionItem(id="high", label="High"),
                OptionItem(id="critical", label="Critical")
            ]
        )
    ]
)

form_extension_spec = FormExtensionSpec(form_render)

agent_detail_extension_spec = AgentDetailExtensionSpec(
    params=AgentDetail(
        interaction_mode="single-turn",
        user_greeting="I'll help you create a well-structured GitHub issue. Fill out the form below.",
        version="0.0.1",
        framework="BeeAI",
        author={
            "name": "Jenna Winkler"
        }
    )
)

@server.agent(
    name="GitHub Issue Writer",
    version="0.0.1",
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    capabilities=a2a.types.AgentCapabilities(
        streaming=True,
        push_notifications=False,
        state_transition_history=False,
        extensions=[
            *form_extension_spec.to_agent_card_extensions(),
            *agent_detail_extension_spec.to_agent_card_extensions(),
        ],
    ),
    skills=[
        AgentSkill(
            id="github_issue_writer",
            name="GitHub Issue Writer",
            description="Create well-structured GitHub issues with proper formatting",
            tags=["GitHub", "Product Management"],
        )
    ],
)
async def github_issue_writer(
    input: Message, 
    context: RunContext,
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer, 
        LLMServiceExtensionSpec.single_demand(
            suggested=("ibm/granite-3-3-8b-instruct", "llama3.1", "gpt-4o-mini")
        )
    ],
    form: Annotated[FormExtensionServer, form_extension_spec]
):
    """This agent provides a structured workflow for generating GitHub issues from user input. Users fill out a form specifying the issue title, type, description, and priority. The agent leverages the BeeAI Framework for managing agent execution, memory, and tool-based reasoning, and uses the BeeAI SDK to expose server endpoints, form-based UI extensions, and trajectory/citation tracking."""
    
    try:
        form_data = form.parse_form_response(message=input)
        values = form_data.values
        
        title = values['title'].value if 'title' in values else 'Untitled Issue'
        issue_types = values['issue_type'].value if 'issue_type' in values else ['feature']
        description = values['description'].value if 'description' in values else ''
        priority = values['priority'].value if 'priority' in values else ['medium']
        
        yield trajectory.trajectory_metadata(
            title="Processing Input",
            content="Analyzing form data and preparing to enhance with AI"
        )
        
        if llm and llm.data:
            llm_config = llm.data.llm_fulfillments.get("default")
            if llm_config:
                llm_client = OpenAIChatModel(
                    model_id=llm_config.api_model,
                    base_url=llm_config.api_base,
                    api_key=llm_config.api_key,
                    tool_choice_support=set(),
                )
                
                memory = UnconstrainedMemory()
                
                tools = [ThinkTool()]
                requirements = [ConditionalRequirement(ThinkTool, force_at_step=1)]
                
                prompt = f"""Transform this into a professional GitHub issue:

Title: {title}
Type: {', '.join(issue_types)}
Priority: {', '.join(priority)}
Description: {description}

Create a complete GitHub issue with improved title, detailed description, appropriate sections, and acceptance criteria. Use proper markdown formatting."""
                
                await memory.add(UserMessage(prompt))
                
                agent = RequirementAgent(
                    llm=llm_client, 
                    memory=memory,
                    tools=tools,
                    requirements=requirements,
                    instructions="Create professional GitHub issues with proper markdown formatting. Be thorough and complete."
                )
                
                yield trajectory.trajectory_metadata(
                    title="Enhancing with AI",
                    content="Using AI to create professional GitHub issue"
                )
                
                full_response = ""
                
                async for event, meta in agent.run(
                    prompt,
                    execution=AgentExecutionConfig(
                        max_iterations=10, 
                        max_retries_per_step=3,
                        total_max_retries=5
                    ),
                    expected_output="Complete GitHub issue in markdown format"
                ):
                    if meta.name == "success" and event.state.steps:
                        step = event.state.steps[-1]
                        if step.tool and step.tool.name == "final_answer":
                            response = step.input.get("response", "")
                            if response and len(response) > 10:
                                full_response = response
                        elif step.tool and step.tool.name == "think":
                            yield trajectory.trajectory_metadata(
                                title="Thinking",
                                content="AI analyzing and structuring the issue"
                            )
                
                if full_response and len(full_response) > 20:
                    yield full_response
                    return
                
                messages = memory.messages
                for msg in reversed(messages):
                    if hasattr(msg, 'role') and msg.role == 'assistant':
                        content = getattr(msg, 'text', None) or getattr(msg, 'content', None)
                        if content and len(content) > 20:
                            yield content
                            return

    except Exception as e:
        yield trajectory.trajectory_metadata(
            title="Error",
            content=f"Error processing form: {str(e)}"
        )
        yield f"Error creating GitHub issue: {str(e)}"

def run():
    """Start the server"""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()
