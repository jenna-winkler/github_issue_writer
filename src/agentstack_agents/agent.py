import os
from typing import Annotated

from dotenv import load_dotenv

import a2a.types
from a2a.types import AgentSkill, Message
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.a2a.extensions import AgentDetailExtensionSpec, AgentDetail, CitationExtensionServer, CitationExtensionSpec, TrajectoryExtensionServer, TrajectoryExtensionSpec, LLMServiceExtensionServer, LLMServiceExtensionSpec
from agentstack_sdk.a2a.extensions.common.form import TextField, SingleSelectField, OptionItem, FormRender
from agentstack_sdk.a2a.extensions.services.form import FormServiceExtensionServer, FormServiceExtensionSpec

from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.think import ThinkTool
from beeai_framework.errors import FrameworkError

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
            id="title",
            label="Issue Title",
            placeholder="Brief, descriptive title for the issue",
            required=True,
            col_span=1
        ),
        SingleSelectField(
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
            id="description",
            label="Description",
            placeholder="What is the issue? What needs to be done?",
            required=True,
            col_span=1
        ),
        SingleSelectField(
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

form_extension_spec = FormServiceExtensionSpec.demand(initial_form=form_render)

agent_detail_extension_spec = AgentDetailExtensionSpec(
    params=AgentDetail(
        interaction_mode="single-turn",
        user_greeting="I'll help you create a well-structured GitHub issue. Fill out the form below.",
        version="0.0.1",
        framework="BeeAI Framework",
        author={
            "name": "Jenna Winkler"
        }
    )
)

@server.agent(
    name="GitHub Issue Writer",
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
            suggested=("ibm-granite/granite-4.0-h-small",)
        )
    ],
    form: Annotated[FormServiceExtensionServer, form_extension_spec]
):
    """This agent provides a structured workflow for generating GitHub issues from user input. Users fill out a form specifying the issue title, type, description, and priority. The agent leverages the BeeAI Framework for managing agent execution, memory, and tool-based reasoning, and uses the Agent Stack SDK to expose server endpoints, form-based UI extensions, and trajectory/citation tracking."""
    
    try:
        # Parse form data
        yield trajectory.trajectory_metadata(
            title="Processing Input",
            content="Parsing form data"
        )
        
        form_data = form.parse_initial_form()
        values = form_data.values
        
        title = values['title'].value if 'title' in values else 'Untitled Issue'
        issue_types = values['issue_type'].value if 'issue_type' in values else ['feature']
        description = values['description'].value if 'description' in values else ''
        priority = values['priority'].value if 'priority' in values else ['medium']
        
        yield trajectory.trajectory_metadata(
            title="Form Data Received",
            content=f"Title: {title} | Type: {''.join(issue_types)} | Priority: {''.join(priority)}"
        )
        
        # Check LLM service availability
        if not llm or not llm.data:
            yield trajectory.trajectory_metadata(
                title="Error",
                content="LLM service extension is required but not available"
            )
            yield "Error: LLM service is not properly configured. Please check your model provider settings in Agent Stack."
            return
        
        # Initialize LLM client
        llm_client = AgentStackChatModel(parameters=ChatModelParameters(stream=True))
        llm_client.set_context(llm)
        
        yield trajectory.trajectory_metadata(
            title="LLM Ready",
            content=f"Using model: {llm_client.model_id}"
        )
        
        # Set up memory and tools
        memory = UnconstrainedMemory()
        tools = [ThinkTool()]
        requirements = [ConditionalRequirement(ThinkTool, force_at_step=1)]
        
        # Create prompt for AI enhancement
        prompt = f"""Transform this into a professional GitHub issue:

Title: {title}
Type: {', '.join(issue_types)}
Priority: {', '.join(priority)}
Description: {description}

Create a complete GitHub issue with:
1. An improved, clear title
2. A detailed description section
3. Relevant sections based on issue type (e.g., Steps to Reproduce for bugs, Use Case for features)
4. Acceptance criteria
5. Proper markdown formatting

Use clear headings (##), bullet points, and formatting to make it professional and easy to read."""
        
        await memory.add(UserMessage(prompt))
        
        # Initialize agent
        agent = RequirementAgent(
            llm=llm_client, 
            memory=memory,
            tools=tools,
            requirements=requirements,
            instructions="Create professional GitHub issues with proper markdown formatting. Be thorough and complete. Use ## for headings, - for bullets, and **bold** for emphasis."
        )
        
        yield trajectory.trajectory_metadata(
            title="Enhancing with AI",
            content="Using AI to create professional GitHub issue"
        )
        
        full_response = ""
        
        # Run agent and stream response
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
                
                if step.tool and step.tool.name == "think":
                    thoughts = step.input.get("thoughts", "Processing...")
                    yield trajectory.trajectory_metadata(
                        title="Thinking",
                        content=thoughts
                    )
                
                elif step.tool and step.tool.name == "final_answer":
                    response = step.input.get("response", "")
                    if response and len(response) > 10:
                        full_response = response
        
        # Return the enhanced issue
        if full_response and len(full_response) > 20:
            yield trajectory.trajectory_metadata(
                title="Complete",
                content="GitHub issue generated successfully"
            )
            yield full_response
            return
        
        # Fallback: check memory for assistant responses
        messages = memory.messages
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role == 'assistant':
                content = getattr(msg, 'text', None) or getattr(msg, 'content', None)
                if content and len(content) > 20:
                    yield trajectory.trajectory_metadata(
                        title="Complete",
                        content="GitHub issue generated successfully"
                    )
                    yield content
                    return
        
        # Final fallback
        yield trajectory.trajectory_metadata(
            title="Warning",
            content="AI did not generate expected output, returning formatted input"
        )
        
        yield f"""## {title}

**Type:** {', '.join(issue_types)}  
**Priority:** {', '.join(priority)}

## Description

{description}

## Acceptance Criteria

- [ ] TODO: Define acceptance criteria
"""

    except FrameworkError as e:
        error_msg = e.explain()
        yield trajectory.trajectory_metadata(
            title="Framework Error",
            content=f"BeeAI Framework error: {error_msg}"
        )
        yield f"Error creating GitHub issue: {error_msg}"
    
    except Exception as e:
        yield trajectory.trajectory_metadata(
            title="Error",
            content=f"Unexpected error: {str(e)}"
        )
        yield f"Error creating GitHub issue: {str(e)}"

def run():
    """Start the server"""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()
    