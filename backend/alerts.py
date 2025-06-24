# alerts.py
import os
import csv
import smtplib
from email.message import EmailMessage
import asyncio

# File to store subscriber emails
SUBSCRIBERS_FILE = "subscribers.csv"

# Confidence threshold for sending alerts
CONFIDENCE_THRESHOLD = 0.75  # send alerts for >=75% confidence


def add_subscriber(email: str) -> bool:
    """
    Add an email to subscribers.csv if not already present.
    Returns True if added, False if already subscribed.
    """
    email = email.strip().lower()
    # Ensure file exists and has header
    if not os.path.isfile(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["email"])
    # Read existing
    with open(SUBSCRIBERS_FILE, mode="r", newline="") as f:
        reader = csv.reader(f)
        subscribers = {row[0].strip().lower() for row in reader if row}
    if email in subscribers:
        return False
    # Append
    with open(SUBSCRIBERS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([email])
    return True


def get_subscribers() -> list:
    """
    Return list of subscribed emails.
    """
    if not os.path.isfile(SUBSCRIBERS_FILE):
        return []
    with open(SUBSCRIBERS_FILE, mode="r", newline="") as f:
        reader = csv.reader(f)
        return [row[0].strip() for row in reader if row and row[0] != "email"]


def _send_email(subject: str, body: str, subscribers: list):
    """Blocking send via smtplib."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port   = int(os.getenv("SMTP_PORT", "587"))
    email_user  = os.getenv("EMAIL_USER")
    email_pass  = os.getenv("EMAIL_PASSWORD")

    if not (smtp_server and email_user and email_pass):
        print("[WARN] SMTP creds missing; skipping email alert.")
        return

    msg = EmailMessage()
    msg["From"]    = email_user
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_user, email_pass)
            for to_email in subscribers:
                msg["To"] = to_email
                server.send_message(msg)
                del msg["To"]
        print(f"[INFO] Alert sent to {len(subscribers)} subscribers.")
    except Exception as e:
        print(f"[ERROR] Sending email failed: {e}")

async def send_email_alert(subject: str, body: str):
    """Async wrapper around the blocking SMTP send."""
    subscribers = get_subscribers()
    if not subscribers:
        return
    # run blocking send in a thread
    await asyncio.to_thread(_send_email, subject, body, subscribers)

async def maybe_alert(confidence: float, token: str, prediction: str):
    """
    If confidence >= threshold, schedule an email alert.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return

    subject = f"EthosX Alert: {token} {prediction} Signal"
    body = (
        f"Signal: {prediction}\n"
        f"Token: {token}\n"
        f"Confidence: {confidence*100:.1f}%\n"
        "Visit your dashboard for details."
    )
    # schedule async send (non‑blocking)
    await send_email_alert(subject, body)