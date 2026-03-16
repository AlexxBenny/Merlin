# skills/email/__init__.py

"""
MERLIN Email Skills Package.

Skills for email compose, edit, send, read, and search.
All skills depend on EmailClient (providers/email/client.py),
never on concrete providers directly.

Skills:
- email.draft_message — Generate email draft via LLM
- email.modify_draft  — Conversational draft editing
- email.send_message  — Send approved draft (destructive)
- email.read_inbox    — Fetch recent email headers
- email.search_email  — Search emails (LLM → IMAP)
"""
