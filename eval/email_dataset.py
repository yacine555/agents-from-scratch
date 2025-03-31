"""Email evaluation dataset with ground truth classifications."""

EMAIL_EVAL_SET = [
    {
        "email_input": {
            "author": "Marketing Team <marketing@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "🔥 Check out our latest newsletter!",
            "email_thread": "Hi everyone,\n\nCheck out this month's company newsletter featuring our latest product updates and team highlights!\n\nDon't forget to subscribe to our social media channels for more updates.\n\nBest regards,\nMarketing Team"
        },
        "ground_truth": "ignore"
    },
    {
        "email_input": {
            "author": "Sarah Johnson <sarah.j@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Team meeting postponed to next week",
            "email_thread": "Hi John,\n\nJust a heads up that our weekly team meeting will be postponed to next Monday due to several team members being out of office.\n\nPlease update your calendar accordingly.\n\nThanks,\nSarah"
        },
        "ground_truth": "notify"
    },
    {
        "email_input": {
            "author": "Michael Chen <m.chen@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Urgent: Help with deployment bug",
            "email_thread": "Hi John,\n\nWe're facing a critical issue with the latest deployment. The authentication service is failing intermittently and users are reporting login problems.\n\nCould you help us troubleshoot this issue? It's affecting our main user flow.\n\nThanks,\nMike"
        },
        "ground_truth": "respond"
    },
    {
        "email_input": {
            "author": "Build System <ci@company.com>",
            "to": "Dev Team <dev-team@company.com>",
            "subject": "Build Failed: main branch (build #4256)",
            "email_thread": "Build Failed: main branch (build #4256)\n\nThe following tests failed:\n- UserAuthenticationTest\n- PaymentProcessingTest\n\nCommit: a8e721f (Fix cart checkout process)\nAuthor: Alex Rivera\n\nSee detailed logs at: https://ci.company.com/builds/4256"
        },
        "ground_truth": "notify"
    },
    {
        "email_input": {
            "author": "Lisa Wong <l.wong@client.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Meeting request - Project Roadmap",
            "email_thread": "Dear John,\n\nI'd like to schedule a meeting to discuss the roadmap for our collaboration on Project Falcon.\n\nWould you be available sometime next week? I'm flexible on Tuesday or Thursday afternoon.\n\nBest regards,\nLisa Wong\nClient Success Manager\nABC Client Inc."
        },
        "ground_truth": "respond"
    }
]