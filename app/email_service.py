import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger
from app.config import settings


def send_invite_email(
    to_email: str,
    inviter_name: str,
    company_name: str,
    data_room_id: str,
):
    """Send an email notification to an invited team member."""
    if not settings.smtp_host or not settings.smtp_user or not settings.smtp_password:
        logger.warning("SMTP not configured — skipping invite email")
        return

    link = f"{settings.frontend_url}/data-room/{data_room_id}"

    subject = f"{inviter_name} invited you to review {company_name}"

    html = f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f9fafb; padding: 40px 0;">
  <div style="max-width: 520px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 36px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
    <h2 style="margin: 0 0 16px; color: #111827; font-size: 20px;">You've been invited to a data room</h2>
    <p style="color: #374151; line-height: 1.6; margin: 0 0 12px;">
      <strong>{inviter_name}</strong> has invited you to collaborate on the
      <strong>{company_name}</strong> data room on Noesis.
    </p>
    <p style="color: #374151; line-height: 1.6; margin: 0 0 24px;">
      Sign in with <strong>{to_email}</strong> to view the data room and start asking questions.
    </p>
    <a href="{link}"
       style="display: inline-block; background: linear-gradient(135deg, #9d174d, #be185d); color: #fff;
              padding: 12px 28px; border-radius: 8px; text-decoration: none; font-weight: 500;">
      Open Data Room
    </a>
    <p style="color: #9ca3af; font-size: 13px; margin: 24px 0 0; line-height: 1.5;">
      If you weren't expecting this invite you can safely ignore this email.
    </p>
  </div>
</body>
</html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, to_email, msg.as_string())
        logger.info(f"Invite email sent to {to_email} for data room {company_name}")
    except Exception as e:
        logger.error(f"Failed to send invite email to {to_email}: {e}")
