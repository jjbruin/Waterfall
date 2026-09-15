"""Email utilities for authentication workflows.

TWO SENDERS, ONE CONTRACT. ``send_email_result`` returns the same
{ok, status, reason} dict whichever service actually carries the message, so
every caller -- welcome, password reset, password changed, feedback replies --
is untouched by which one is configured. Azure Communication Services wins
when it is configured, SendGrid is the fallback.

WHY WE MOVED. SendGrid retired its free plan in 2025; ours lapsed and from
~Aug 1 2026 every outbound email failed with "Maximum credits exceeded" -- a
message that reads as "you sent too much" on an account that had sent nothing.
ACS bills to the Azure subscription this app already runs in, at $0.00025 per
email, which at our volume is cents a year.

KEEPING SENDGRID IS DELIBERATE. The cutover is an env-var change and so is the
rollback, with no rebuild in either direction. Delete the SendGrid path once
ACS has carried real mail for a while, not before.
"""

import json
import logging
import requests
from flask import current_app

log = logging.getLogger(__name__)

# Sending is bounded so a slow provider cannot hold a web request open. ACS
# accepts the message and then reports delivery asynchronously; we wait only
# long enough to catch the failures an admin can act on (bad sender, bad
# credentials, unverified domain) and treat "still sending" as sent, because
# it is -- ACS has taken custody of the message by then.
_ACS_WAIT_SECONDS = 20


def _acs_configured() -> bool:
    return bool(current_app.config.get("ACS_CONNECTION_STRING")
                and current_app.config.get("ACS_SENDER"))


def _explain_acs(exc: Exception) -> str:
    """Turn an ACS exception into something an admin can act on.

    The two failures that actually happen in practice are an unverified or
    mistyped sender address and a bad connection string, and ACS reports both
    as an HttpResponseError whose useful detail is buried in the message. Say
    which one it looks like rather than echoing a stack trace into a toast.
    """
    text = str(exc)
    low = text.lower()
    if "domain" in low and ("not" in low and "verif" in low):
        return ("Azure rejected the sender domain as unverified. Finish the DNS "
                "verification (TXT, SPF and the DKIM CNAMEs) on the Email "
                "Communication Service before sending.")
    if "senderaddress" in low.replace(" ", "") or "sender" in low:
        return (f"Azure rejected the sender address "
                f"'{current_app.config.get('ACS_SENDER', '')}'. It must be an "
                f"address on a domain connected to the Communication Service.")
    if "unauthorized" in low or "401" in text or "403" in text:
        return ("Azure rejected the credentials. Check ACS_CONNECTION_STRING — "
                "it is the whole connection string, endpoint and key together.")
    return f"Azure Communication Services error: {text[:200]}"


def _send_via_acs(to: str, subject: str, html_body: str) -> dict:
    """Send through Azure Communication Services."""
    try:
        from azure.communication.email import EmailClient
    except ImportError:
        return {"ok": False, "status": None,
                "reason": "The azure-communication-email package is not "
                          "installed in this image."}

    sender = current_app.config["ACS_SENDER"]
    try:
        client = EmailClient.from_connection_string(
            current_app.config["ACS_CONNECTION_STRING"])
        poller = client.begin_send({
            "senderAddress": sender,
            "recipients": {"to": [{"address": to}]},
            "content": {"subject": subject, "html": html_body},
        })
    except Exception as e:
        log.error("ACS rejected the send to %s: %s", to, e)
        return {"ok": False, "status": None, "reason": _explain_acs(e)}

    # The message is ACS's problem now. Wait briefly for a terminal status so
    # a genuine rejection is reported while the admin is still looking at the
    # screen, but do not fail a message that is merely still in flight.
    try:
        poller.wait(timeout=_ACS_WAIT_SECONDS)
    except Exception as e:
        log.warning("ACS send to %s accepted but polling failed: %s", to, e)
        return {"ok": True, "status": "Accepted", "reason": None}

    if not poller.done():
        log.info("Email to %s accepted by ACS, still sending: %s", to, subject)
        return {"ok": True, "status": "Running", "reason": None}

    try:
        result = poller.result() or {}
    except Exception as e:
        log.error("ACS send to %s failed: %s", to, e)
        return {"ok": False, "status": None, "reason": _explain_acs(e)}

    status = result.get("status") or ""
    if status.lower() in ("succeeded", "running", "notstarted"):
        log.info("Email sent to %s via ACS (%s): %s", to, status, subject)
        return {"ok": True, "status": status, "reason": None}

    detail = result.get("error") or {}
    message = detail.get("message") if isinstance(detail, dict) else str(detail)
    log.error("ACS reported %s sending to %s: %s", status, to, message)
    return {"ok": False, "status": status,
            "reason": f"Azure could not deliver the message ({status})."
                      + (f" {message}" if message else "")}


def _explain(status: int, body: str) -> str:
    """Turn SendGrid's wording into something an admin can act on.

    "Maximum credits exceeded" reads as "you have sent too much", and on
    Sep 15 2026 that sent us looking at usage for an account that had sent
    NOTHING: /v3/user/credits returned total 0, used 0, with the daily reset
    stuck at 2026-08-01. The allowance itself was zero -- the free plan had
    lapsed. The message has to say that, or the next person debugs the wrong
    thing for an hour as well.
    """
    low = (body or "").lower()
    if "maximum credits exceeded" in low:
        return ("SendGrid reports no sending credits on the account. This is a "
                "plan/billing issue at SendGrid, not a problem with the app or "
                "the recipient — no email can be sent until the account has an "
                "allowance again.")
    if status in (401, 403):
        return f"SendGrid rejected the API key ({status}). Check SENDGRID_API_KEY."
    if status == 413:
        return "The message was too large for SendGrid to accept."
    detail = (body or "").strip()
    return f"SendGrid returned {status}" + (f": {detail[:200]}" if detail else "")


def send_email_result(to: str, subject: str, html_body: str) -> dict:
    """Send via SendGrid and say WHY if it failed.

    send_email() keeps its boolean contract for the callers that only branch
    on success; anything that has to tell a human what went wrong uses this.
    """
    if _acs_configured():
        return _send_via_acs(to, subject, html_body)

    api_key = current_app.config.get("SENDGRID_API_KEY", "")
    from_email = current_app.config.get("SENDGRID_FROM", "")
    if not api_key or not from_email:
        log.warning("Email not configured — message to %s not sent: %s", to, subject)
        return {"ok": False, "status": None,
                "reason": "Email is not configured on this deployment. Set "
                          "ACS_CONNECTION_STRING and ACS_SENDER (Azure "
                          "Communication Services), or SENDGRID_API_KEY and "
                          "SENDGRID_FROM."}

    try:
        resp = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "personalizations": [{"to": [{"email": to}]}],
                "from": {"email": from_email},
                "subject": subject,
                "content": [{"type": "text/html", "value": html_body}],
            },
            timeout=15,
        )
        if resp.status_code in (200, 202):
            log.info("Email sent to %s: %s", to, subject)
            return {"ok": True, "status": resp.status_code, "reason": None}
        log.error("SendGrid error %s sending to %s: %s",
                  resp.status_code, to, resp.text)
        return {"ok": False, "status": resp.status_code,
                "reason": _explain(resp.status_code, resp.text)}
    except Exception as e:
        log.error("Failed to send email to %s: %s", to, e)
        return {"ok": False, "status": None,
                "reason": f"Could not reach SendGrid: {str(e)[:200]}"}


def send_email(to: str, subject: str, html_body: str) -> bool:
    """Send an email via SendGrid API. Returns True on success."""
    return send_email_result(to, subject, html_body)["ok"]


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


def send_welcome_email(email: str, username: str, temp_password: str) -> dict:
    """Send a welcome email with login credentials.

    Returns the detailed result, not a bare bool: an admin who has just
    created a user needs to be told WHY the mail did not go and that the
    account works anyway.
    """
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
        <p style="font-weight: 600; margin-bottom: 8px;">Add a desktop shortcut:</p>
        <ol style="font-size: 13px; color: #444; padding-left: 20px; margin-top: 0;">
            <li>Click the button below to download the setup file</li>
            <li>Open the downloaded file (double-click it)</li>
            <li>A "Waterfall XIRR" shortcut will appear on your desktop</li>
        </ol>
        <p style="margin: 16px 0;">
            <a href="{app_url}/auth/shortcut/install"
               style="background: #548235; color: white; padding: 10px 20px;
                      text-decoration: none; border-radius: 6px; display: inline-block;
                      font-size: 13px;">
                Download Desktop Shortcut Setup
            </a>
        </p>
    </div>
    """
    return send_email_result(
        email, "Welcome to Waterfall XIRR — Your Account Is Ready", html)


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
