"""Email utilities for authentication workflows."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app

log = logging.getLogger(__name__)


def _get_smtp_config() -> dict:
    cfg = current_app.config
    return {
        "host": cfg.get("SMTP_HOST", "smtp.office365.com"),
        "port": cfg.get("SMTP_PORT", 587),
        "user": cfg.get("SMTP_USER", ""),
        "password": cfg.get("SMTP_PASSWORD", ""),
        "from_addr": cfg.get("SMTP_FROM") or cfg.get("SMTP_USER", ""),
    }


def send_email(to: str, subject: str, html_body: str) -> bool:
    """Send an email. Returns True on success."""
    smtp = _get_smtp_config()
    if not smtp["user"] or not smtp["password"]:
        log.warning("SMTP not configured — email to %s not sent: %s", to, subject)
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp["from_addr"]
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp["host"], smtp["port"], timeout=15) as server:
            server.starttls()
            server.login(smtp["user"], smtp["password"])
            server.send_message(msg)
        log.info("Email sent to %s: %s", to, subject)
        return True
    except Exception as e:
        log.error("Failed to send email to %s: %s", to, e)
        return False


def send_password_reset_email(email: str, username: str, reset_token: str) -> bool:
    """Send a password reset link."""
    app_url = current_app.config.get("APP_URL", "")
    reset_link = f"{app_url}/reset-password?token={reset_token}"
    login_link = f"{app_url}/login"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto;">
        <h2 style="color: #1d4e7e;">Waterfall XIRR — Password Reset</h2>
        <p>Hi <strong>{username}</strong>,</p>
        <p>We received a request to reset your password. Click the link below to set a new password:</p>
        <p style="margin: 24px 0;">
            <a href="{reset_link}"
               style="background: #1d4e7e; color: white; padding: 12px 24px;
                      text-decoration: none; border-radius: 6px; display: inline-block;">
                Reset Password
            </a>
        </p>
        <p style="font-size: 13px; color: #666;">
            This link expires in 1 hour. If you didn't request this, you can safely ignore this email.
        </p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;" />
        <p style="font-size: 13px; color: #666;">
            <a href="{login_link}">Log in to Waterfall XIRR</a>
        </p>
    </div>
    """
    return send_email(email, "Waterfall XIRR — Password Reset", html)


def send_welcome_email(email: str, username: str, temp_password: str) -> bool:
    """Send a welcome email with login credentials."""
    app_url = current_app.config.get("APP_URL", "")
    login_link = f"{app_url}/login"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto;">
        <h2 style="color: #1d4e7e;">Welcome to Waterfall XIRR</h2>
        <p>Hi <strong>{username}</strong>,</p>
        <p>Your account has been created. Here are your login credentials:</p>
        <div style="background: #f5f7fa; border-radius: 8px; padding: 16px; margin: 16px 0;">
            <table style="font-size: 14px;">
                <tr><td style="color: #666; padding-right: 12px;">Username:</td><td><strong>{username}</strong></td></tr>
                <tr><td style="color: #666; padding-right: 12px;">Password:</td><td><strong>{temp_password}</strong></td></tr>
            </table>
        </div>
        <p>You will be asked to set a new password when you log in for the first time.</p>
        <p style="margin: 24px 0;">
            <a href="{login_link}"
               style="background: #1d4e7e; color: white; padding: 12px 24px;
                      text-decoration: none; border-radius: 6px; display: inline-block;">
                Log In to Waterfall XIRR
            </a>
        </p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;" />
        <p style="font-size: 13px; color: #666;">
            Tip: After logging in, you can save a shortcut to your desktop by
            dragging the URL from your browser's address bar to your desktop.
        </p>
    </div>
    """
    return send_email(email, "Welcome to Waterfall XIRR — Your Account Is Ready", html)


def send_password_changed_email(email: str, username: str) -> bool:
    """Send a confirmation that the password was changed."""
    app_url = current_app.config.get("APP_URL", "")
    login_link = f"{app_url}/login"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto;">
        <h2 style="color: #1d4e7e;">Waterfall XIRR — Password Changed</h2>
        <p>Hi <strong>{username}</strong>,</p>
        <p>Your password has been successfully changed.</p>
        <p>If you did not make this change, please contact your administrator immediately.</p>
        <p style="margin: 24px 0;">
            <a href="{login_link}"
               style="background: #1d4e7e; color: white; padding: 12px 24px;
                      text-decoration: none; border-radius: 6px; display: inline-block;">
                Log In
            </a>
        </p>
    </div>
    """
    return send_email(email, "Waterfall XIRR — Password Changed", html)
