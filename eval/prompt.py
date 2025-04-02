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

RESPONSE_QUALITY_PROMPT = """

<Task>
You are evaluating the quality of the email response.

If a response is warranted, it should match the tone and style of the reference response.

If the decision is to ignore or notify, then you can confirm that the decision is correct.

You will be given:
- the email_input
- the agent's reasoning and response as a list of messages 
- the reference response

Your job is to evaluate the agent's reasoning, decision, and response relative to the reference response.
</Task>

<email_input>
{inputs}
</email_input>

<agent_response>
{outputs}
</agent_response>

<correct_response>
{reference_outputs}
</correct_response>
"""
