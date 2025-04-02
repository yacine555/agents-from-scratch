from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated, Optional
from langgraph.graph import add_messages

class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

class State(TypedDict):
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]
    messages: Annotated[list, add_messages]
    use_semantic_memory: bool

# Define schemas for Agent Inbox integration
class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool

class ActionRequest(TypedDict):
    action: str
    args: dict

class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: Optional[str]

# TODO: Load into memory 
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}