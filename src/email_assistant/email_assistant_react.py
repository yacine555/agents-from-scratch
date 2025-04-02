from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.email_assistant.prompts import agent_system_prompt_react, prompt_instructions
from src.email_assistant.schemas import profile

from langgraph.graph import MessagesState, StateGraph, START, END

# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

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

@tool
def triage_email(category: Literal["ignore", "notify", "respond"]) -> str:
    """Triage an email into one of three categories: ignore, notify, respond."""
    return f"Classification Decision: {category}"

def create_prompt(state):
    return [
        {"role": "system", "content": agent_system_prompt_react.format(instructions=prompt_instructions["agent_instructions"], **profile)}
    ] + state['messages']

# Agent tools default
tools = [write_email, schedule_meeting, check_calendar_availability, triage_email]

# Agent prompt default
prompt = agent_system_prompt_react

# Create agent
agent = create_react_agent(
    llm,
    tools=tools,
    prompt=prompt,
)

# Build workflow
email_assistant_react = (
    StateGraph(MessagesState)
    .add_node("response_agent", agent)
    .add_edge(START, "response_agent")
    .compile()
)