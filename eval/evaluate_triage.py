from langsmith import Client
from langsmith import testing as t

from openevals.llm import create_llm_as_judge
from eval.prompt import TRIAGE_CLASSIFICATION_PROMPT

from src.email_assistant.email_assistant import email_assistant
from src.email_assistant.email_assistant_react import email_assistant_react 

## Client 
client = Client()

## Dataset 
# Programmatically create a dataset in LangSmith
dataset = client.create_dataset(dataset_name="Interrupt Workshop: E-mail Triage Dataset", 
                                description="A dataset of e-mails and their triage decisions."
)

# Dataset examples
# TODO: Add more examples
email_input_1 = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

# Outputs options "ignore", "notify", "respond"
triage_output_1 = "respond"

examples = [
  {
      "inputs": {"email_input": email_input_1},
      "outputs": {"classification": triage_output_1},
  },
  {
      "inputs": {"email_input": " TODO "},
      "outputs": {"classification": " TODO "},
  },
]

# Add examples to the dataset
client.create_examples(dataset_id=dataset.id, examples=examples)

## Functions
def target_email_assistant(inputs: dict) -> dict:
    """ Take input dict from each row in examples, extract email_input, pass to the email assistant, return the response """
    response = email_assistant.invoke({"email_input": inputs["email_input"]})
    return { "response": response['messages'] }

def target_email_assistant_react(inputs: dict) -> dict:
    """ Take input dict from each row in examples, extract email_input, pass to the react version of the email assistant, return the response """
    messages = [{"role": "user", "content": str(inputs["email_input"])}]
    response = email_assistant_react.invoke({"messages": messages})
    return { "response": response['messages'] }

## Evaluator 
conciseness_evaluator = create_llm_as_judge(
    prompt=TRIAGE_CLASSIFICATION_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini",
)

## Run evaluation
experiment_results_react = client.evaluate(
    # Function that takes input dict from each row in examples, extract email_input, pass to the react version of the email assistant, return the response 
    inputs=target_email_assistant_react,
    # Dataset name defined above 
    data="Interrupt Workshop: E-mail Triage Dataset",
    # Pre-built evaluator which passes:
    #  inputs from dataset to "inputs" key in TRIAGE_CLASSIFICATION_PROMPT
    #  outputs from target fxn (target_email_assistant_react) to "outputs" key in TRIAGE_CLASSIFICATION_PROMPT
    #  reference_outputs from dataset to "reference_outputs" key in TRIAGE_CLASSIFICATION_PROMPT
    evaluators=[
        conciseness_evaluator
    ],
    experiment_prefix="E-mail assistant react", # Name of the experiment
    max_concurrency=2, # Number of concurrent evaluations
)

experiment_results_workflow = client.evaluate(
    # Function that takes input dict from each row in examples, extract email_input, pass to the react version of the email assistant, return the response 
    inputs=email_assistant,
    # Dataset name defined above 
    data="Interrupt Workshop: E-mail Triage Dataset",
    # Pre-built evaluator which passes:
    #  inputs from dataset to "inputs" key in TRIAGE_CLASSIFICATION_PROMPT
    #  outputs from target fxn (target_email_assistant_react) to "outputs" key in TRIAGE_CLASSIFICATION_PROMPT
    #  reference_outputs from dataset to "reference_outputs" key in TRIAGE_CLASSIFICATION_PROMPT
    evaluators=[
        conciseness_evaluator
    ],
    experiment_prefix="E-mail assistant workflow", # Name of the experiment
    max_concurrency=2, # Number of concurrent evaluations
)


 



