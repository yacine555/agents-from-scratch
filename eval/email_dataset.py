"""Email evaluation dataset with ground truth classifications."""

# Dataset examples
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

email_input_2 = {
    "author": "Marketing Team <marketing@company.com>",
    "to": "All Staff <all-staff@company.com>",
    "subject": "New Company Newsletter Available",
    "email_thread": """Hello everyone,

The latest edition of our company newsletter is now available on the intranet. This month features articles on our Q2 results, upcoming team building activities, and employee spotlights.

Check it out when you have a chance!

Best regards,
Marketing Team""",
}

email_input_3 = {
    "author": "System Admin <sysadmin@company.com>",
    "to": "Development Team <dev@company.com>",
    "subject": "Scheduled maintenance - database downtime",
    "email_thread": """Hi team,

This is a reminder that we'll be performing scheduled maintenance on the production database tonight from 2AM to 4AM EST. During this time, all database services will be unavailable.

Please plan your work accordingly and ensure no critical deployments are scheduled during this window.

Thanks,
System Admin Team""",
}

email_input_4 = {
    "author": "Project Manager <pm@client.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "URGENT: Critical bug in production",
    "email_thread": """John,

We've discovered a critical bug in the payment processing module that's causing transactions to fail for some customers. This is affecting our revenue and causing customer complaints.

Can you please look into this ASAP and provide an estimate for a fix? This is our highest priority right now.

Regards,
Project Manager""",
}

email_input_5 = {
    "author": "HR Department <hr@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Reminder: Submit your expense reports",
    "email_thread": """Hello John,

This is a friendly reminder that all expense reports for the previous month need to be submitted by this Friday. Please make sure to include all receipts and proper documentation.

If you have any questions about the submission process, feel free to reach out to the HR team.

Best regards,
HR Department""",
}

# Triage outputs: "ignore", "notify", "respond"
triage_output_1 = "respond"
triage_output_2 = "ignore"
triage_output_3 = "notify"
triage_output_4 = "respond"
triage_output_5 = "notify"

# End-to-end response outputs (when applicable)
response_output_1 = "I've updated the docs with the missing endpoints. Let me know if you need anything else."
response_output_2 = "No response is needed. For evaluation purposes, just confirm that the agent is to ignore."
response_output_3 = "No response is needed. For evaluation purposes, just confirm that the agent is to notify."
response_output_4 = "I've fixed the bug! Let me know if you need anything else."
response_output_5 = "No response is needed. For evaluation purposes, just confirm that the agent is to notify."

examples_triage = [
  {
      "inputs": {"email_input": email_input_1},
      "outputs": {"classification": triage_output_1},
  },
  {
      "inputs": {"email_input": email_input_2},
      "outputs": {"classification": triage_output_2},
  },
  {
      "inputs": {"email_input": email_input_3},
      "outputs": {"classification": triage_output_3},
  },
  {
      "inputs": {"email_input": email_input_4},
      "outputs": {"classification": triage_output_4},
  },
  {
      "inputs": {"email_input": email_input_5},
      "outputs": {"classification": triage_output_5},
  },
]

examples_response = [
  {
      "inputs": {"email_input": email_input_1},
      "outputs": {"response": response_output_1},
  },
  {
      "inputs": {"email_input": email_input_2},
      "outputs": {"response": response_output_2},
  },
  {
      "inputs": {"email_input": email_input_3},
      "outputs": {"response": response_output_3},
  },
  {
      "inputs": {"email_input": email_input_4},
      "outputs": {"response": response_output_4},
  },
  {
      "inputs": {"email_input": email_input_5},
      "outputs": {"response": response_output_5},
  },
]
