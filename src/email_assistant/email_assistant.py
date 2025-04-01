from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt, agent_system_prompt_memory, prompt_instructions
from email_assistant.schemas import State, RouterSchema, profile
from email_assistant.utils import parse_email
from email_assistant.configuration import Configuration
from langmem import create_manage_memory_tool, create_search_memory_tool

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# We'll use structured output to generate classification results
llm_router = llm.with_structured_output(RouterSchema) 

# Agent tools 
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

def create_prompt(state):
    return [
        {"role": "system", "content": agent_system_prompt.format(instructions=prompt_instructions["agent_instructions"], **profile)}
    ] + state['messages']

# Agent tools default
tools = [write_email, schedule_meeting, check_calendar_availability]

# Agent prompt default
prompt = agent_system_prompt

# Append to our tools based on configuration 
config = Configuration()
if config.use_semantic_memory: # TODO: Is this correct?
    # Create semantic memory tools
    manage_memory_tool = create_manage_memory_tool(namespace=("email_assistant", "{langgraph_user_id}", "collection"))
    search_memory_tool = create_search_memory_tool(namespace=("email_assistant", "{langgraph_user_id}", "collection"))
    # Add semantic memory tools
    tools.append(manage_memory_tool, search_memory_tool)
    # Update prompt
    prompt = agent_system_prompt_memory

# Create agent
agent = create_react_agent(
    llm,
    tools=tools,
    prompt=prompt,
)

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
    )

    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [{
                    "role": "user",
                    "content": f"Classification Decision: {result.classification}",
                },
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ],
            "classification_decision": result.classification,
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update =  {
            "messages": [
                {
                    "role": "user",
                    "content": f"Classification Decision: {result.classification}"
                }
            ],
            "classification_decision": result.classification,
        }
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Classification Decision: {result.classification}"
                }
            ],
            "classification_decision": result.classification,
        }
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

# Build workflow
agent = (
    StateGraph(State)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
    .compile()
)