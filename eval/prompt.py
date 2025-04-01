TRIAGE_CLASSIFICATION_PROMPT = """

<Task>
You are evaluating the classification of emails.

They should be be classified into one of the following categories:
- ignore
- notify
- respond

You will be given:
- the email_input
- the agent's reasoning and decision as a list of messages 
- the correct classification

Your job is to evaluate the agent's reasoning and decision relative to the correct classification.
</Task>

<email_input>
{inputs}
</email_input>

<agent_response>
{outputs}
</agent_response>

<correct_classification>
{reference_outputs}
</correct_classification>
"""
