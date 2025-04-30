import os, ssl, smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()                                  # loads .env

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)

def send_email(recipient: str, subject: str, body: str) -> None:
    """Send a plain-text e-mail; raises if creds are missing or wrong."""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS]):
        raise RuntimeError("SMTP credentials not set")

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = EMAIL_FROM, recipient, subject
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=10) as s:
        # For port 587 use: s = smtplib.SMTP(SMTP_HOST, 587); s.starttls(context=context)
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)