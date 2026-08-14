"""
Persisted, Super-Admin-configurable email sender credentials — Gmail or
Microsoft Outlook/Office 365 (2026-08-14: the team mainly uses Outlook for
mail, but wants Gmail kept available too since either might end up sending).

Same pattern as router.py's runtime backend settings: a small JSON file on
disk, loaded once at import and re-writable at runtime, so the team running
this deployment can rotate the provider, sender address, or app password
from the Super Admin panel — no .env edit, no code change, no container
restart. Confirmed live (2026-08-14): a `docker restart` alone does NOT
pick up an edited .env (env_file values are baked in at container
creation, not re-read on restart) — this module sidesteps that whole class
of problem by reading its own state file fresh on every call instead.

Falls back to the GMAIL_SENDER / GMAIL_APP_PASSWORD env vars (provider
"gmail") until an admin has saved a value here at least once, so an
existing .env-based setup keeps working unchanged after this ships.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_STATE_DIR = os.getenv("API_STATE_DIR", os.path.join(os.path.dirname(__file__), "state"))
_SETTINGS_PATH = os.path.join(_STATE_DIR, "email_settings.json")

# host/port/security mode for each supported provider — smtp_utils.py's
# connect helper switches on these, not on the provider name directly, so
# adding a third provider later only means adding one entry here.
PROVIDERS = {
    "gmail": {
        "label": "Gmail",
        "host": "smtp.gmail.com",
        "port": 465,
        "mode": "ssl",       # smtplib.SMTP_SSL
    },
    "outlook": {
        "label": "Microsoft Outlook / Office 365",
        "host": "smtp.office365.com",
        "port": 587,
        "mode": "starttls",  # smtplib.SMTP + .starttls()
    },
}
_DEFAULT_PROVIDER = "gmail"

_runtime_provider = ""
_runtime_sender = ""
_runtime_app_password = ""


def _load() -> None:
    global _runtime_provider, _runtime_sender, _runtime_app_password
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _runtime_provider = data.get("provider", "")
        _runtime_sender = data.get("sender", "")
        _runtime_app_password = data.get("app_password", "")
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[email_settings] failed to load email_settings.json: %s", exc)


def _save() -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    data = {
        "provider": _runtime_provider,
        "sender": _runtime_sender,
        "app_password": _runtime_app_password,
    }
    tmp_path = f"{_SETTINGS_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, _SETTINGS_PATH)


_load()


def get_provider() -> str:
    """"gmail" or "outlook" — which SMTP host/port/security mode to use.
    Falls back to the EMAIL_PROVIDER env var, then "gmail" (the only
    provider this app spoke before 2026-08-14), so nothing already
    running breaks when this ships."""
    p = _runtime_provider or os.getenv("EMAIL_PROVIDER", "").strip().lower()
    return p if p in PROVIDERS else _DEFAULT_PROVIDER


def get_sender() -> str:
    """Address escalation/quotation emails are sent FROM. Re-reads the
    in-memory value on every call (no caching) — a panel save takes effect
    on the very next email, mid-session, no restart."""
    return _runtime_sender or os.getenv("GMAIL_SENDER", "") or os.getenv("OUTLOOK_SENDER", "")


def get_app_password() -> str:
    """The mailbox's App Password (Gmail and Microsoft both use this exact
    term for the same idea) used to authenticate the SMTP login — NOT the
    account's normal login password. Both providers reject a normal
    password for SMTP once multi-factor auth is on, which is required to
    even generate an App Password in the first place."""
    return (
        _runtime_app_password
        or os.getenv("GMAIL_APP_PASSWORD", "")
        or os.getenv("OUTLOOK_APP_PASSWORD", "")
    )


def get_email_settings() -> dict:
    """For the Super Admin panel. The password itself is never returned —
    only whether one is currently set (from a panel save or an env var)."""
    return {
        "provider": get_provider(),
        "sender": get_sender(),
        "app_password_set": bool(get_app_password()),
    }


def set_email_settings(provider: str = "", sender: str = "", app_password: str = "") -> dict:
    """provider/sender/app_password are only overwritten when a non-empty
    (and, for provider, recognized) value is passed, so saving one field
    from the panel never requires re-entering the others — same "leave
    blank to keep current" behavior as the backend settings' manual_api_key.
    There's deliberately no way to blank sender/app_password back out from
    here: an empty one would silently mute every future escalation, exactly
    the failure mode this exists to prevent."""
    global _runtime_provider, _runtime_sender, _runtime_app_password
    if provider.strip().lower() in PROVIDERS:
        _runtime_provider = provider.strip().lower()
    if sender.strip():
        _runtime_sender = sender.strip()
    if app_password.strip():
        _runtime_app_password = app_password.strip()
    _save()
    logger.info(
        "[email_settings] provider=%s sender=%s",
        get_provider(), _runtime_sender or "(unset — using env var fallback)",
    )
    return get_email_settings()
