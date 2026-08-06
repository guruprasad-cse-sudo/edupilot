"""
EduPilot authentication module.

Provides multi-user support on top of streamlit-authenticator:
  - users.yaml credential store (bcrypt-hashed passwords; git-ignored)
  - login / registration UI
  - cookie-based session persistence (stays signed in across refreshes)
  - per-user run-history directories (shared knowledge base stays global)

Security notes:
  - The cookie signing key comes from the SESSION_SECRET environment
    variable and is never written to disk.
  - users.yaml holds only credential records (bcrypt hashes), no secrets
    in plaintext, and is git-ignored.
  - Usernames are restricted to a strict allowlist and the per-user runs
    directory is verified to stay inside config.runs_dir.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets as _secrets
import time
from pathlib import Path
from typing import Dict, List

import bcrypt
import streamlit as st
import streamlit_authenticator as stauth
import yaml

from config import config
from logging_utils import get_logger

logger = get_logger(__name__)

USERS_FILE: Path = Path(__file__).resolve().parent / "users.yaml"

_SS_AUTHENTICATOR = "_authenticator"

# Strict allowlist for usernames used as directory names.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,64}$")

# ---------------------------------------------------------------------------
# Faculty password policy
# ---------------------------------------------------------------------------
# Policy rationale:
#   - Min 10 chars: meaningfully stronger than the stauth default of 8,
#     without being burdensome for faculty.
#   - Max 128 chars: supports long passphrases; no artificial 20-char cap.
#   - Upper + lower + digit: clear, standard requirements that block the
#     most common weak patterns.
#   - No mandatory special character: avoids friction for passphrase-style
#     passwords and is not required by NIST 800-63b for this threat model.
_PW_MIN_LEN: int = 10
_PW_MAX_LEN: int = 128
_PW_REQUIRE_UPPER: bool = True
_PW_REQUIRE_LOWER: bool = True
_PW_REQUIRE_DIGIT: bool = True

PASSWORD_POLICY_SUMMARY: str = (
    f"Password must be {_PW_MIN_LEN}–{_PW_MAX_LEN} characters and contain "
    "at least one uppercase letter, one lowercase letter, and one digit."
)

_PW_RE = re.compile(
    rf"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{{{_PW_MIN_LEN},{_PW_MAX_LEN}}}$"
)


class _FacultyPasswordValidator(stauth.utilities.validator.Validator):
    """Custom Validator that enforces EduPilot's faculty password policy."""

    def validate_password(self, password: str) -> bool:
        return bool(_PW_RE.match(password or ""))

    def diagnose_password(self, password: str) -> str:
        """Return a plain-English description of unmet requirements."""
        pw = password or ""
        errors: list[str] = []
        if not (_PW_MIN_LEN <= len(pw) <= _PW_MAX_LEN):
            errors.append(f"be {_PW_MIN_LEN}–{_PW_MAX_LEN} characters long")
        if _PW_REQUIRE_LOWER and not re.search(r"[a-z]", pw):
            errors.append("contain at least one lowercase letter")
        if _PW_REQUIRE_UPPER and not re.search(r"[A-Z]", pw):
            errors.append("contain at least one uppercase letter")
        if _PW_REQUIRE_DIGIT and not re.search(r"\d", pw):
            errors.append("contain at least one digit")
        if not errors:
            return ""
        joined = "; ".join(errors)
        return f"Password must {joined}."

# ---------------------------------------------------------------------------
# Forgot-password rate limiting
# ---------------------------------------------------------------------------

# Module-level store: maps session_id → list of attempt timestamps (monotonic).
# Lives in the server process; resets on restart (acceptable — limits live
# abuse, not archived state).
_FP_ATTEMPTS: Dict[str, List[float]] = {}
_FP_MAX_ATTEMPTS: int = 5
_FP_WINDOW_SECS: int = 15 * 60  # 15 minutes

_GENERIC_RESET_ERROR = "The details provided do not match our records."


def _fp_session_id() -> str:
    """Return a stable identifier for the current Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx and ctx.session_id:
            return ctx.session_id
    except Exception:  # noqa: BLE001
        pass
    # Fallback: ephemeral key stored in session state.
    if "_fp_sid" not in st.session_state:
        st.session_state["_fp_sid"] = _secrets.token_hex(16)
    return st.session_state["_fp_sid"]


def _fp_attempts_in_window(session_id: str) -> int:
    """Return number of forgot-password attempts in the current window."""
    now = time.monotonic()
    attempts = [t for t in _FP_ATTEMPTS.get(session_id, []) if now - t < _FP_WINDOW_SECS]
    _FP_ATTEMPTS[session_id] = attempts  # prune in place
    return len(attempts)


def _fp_record_attempt(session_id: str) -> None:
    """Append a timestamp for a forgot-password submission."""
    _FP_ATTEMPTS.setdefault(session_id, []).append(time.monotonic())


def _cookie_key() -> str:
    """Cookie signing key from the environment (never persisted to disk)."""
    key = os.environ.get("SESSION_SECRET")
    if not key:
        # Fail safe: random per-process key. Sessions won't survive a server
        # restart, but authentication itself still works.
        logger.warning(
            "SESSION_SECRET not set — using an ephemeral cookie key; "
            "signed-in sessions will not survive a server restart."
        )
        key = _secrets.token_hex(32)
        os.environ["SESSION_SECRET"] = key  # stable within this process
    return key


def _load_credentials() -> dict:
    """Load the credential store from users.yaml (empty store if missing)."""
    if USERS_FILE.exists():
        try:
            data = yaml.safe_load(USERS_FILE.read_text(encoding="utf-8")) or {}
            creds = data.get("credentials")
            if isinstance(creds, dict) and isinstance(creds.get("usernames"), dict):
                return creds
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse %s: %s", USERS_FILE, exc)
            raise
    return {"usernames": {}}


def _save_new_user(username: str) -> None:
    """Persist a newly registered user to users.yaml.

    Re-reads the file under an exclusive lock and merges only the new
    username, so concurrent registrations from other sessions are not
    clobbered. Written atomically via temp file + rename.
    """
    session_creds = get_authenticator().authentication_controller.authentication_model.credentials
    new_record = session_creds["usernames"].get(username)
    if new_record is None:
        raise RuntimeError(f"Registered user {username!r} missing from session credentials")

    lock_path = USERS_FILE.with_suffix(".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            on_disk = _load_credentials()
            on_disk["usernames"][username] = new_record
            tmp_path = USERS_FILE.with_suffix(".tmp")
            tmp_path.write_text(
                yaml.safe_dump({"credentials": on_disk}), encoding="utf-8"
            )
            tmp_path.replace(USERS_FILE)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    logger.info("Credential store updated: added %s", username)


def _save_updated_password(username: str) -> None:
    """Persist an updated (reset) password hash to users.yaml.

    Reads the new bcrypt hash from the in-session authenticator credential
    store (which streamlit-authenticator has already updated in memory),
    then writes it atomically under an exclusive lock.  The cookie key and
    any other env-sourced secrets are never written to disk.
    """
    session_creds = get_authenticator().authentication_controller.authentication_model.credentials
    updated_record = session_creds["usernames"].get(username)
    if updated_record is None:
        raise RuntimeError(f"User {username!r} missing from session credentials after reset")

    lock_path = USERS_FILE.with_suffix(".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            on_disk = _load_credentials()
            if username not in on_disk["usernames"]:
                raise RuntimeError(
                    f"Cannot update password: user {username!r} not found in {USERS_FILE}"
                )
            on_disk["usernames"][username] = updated_record
            tmp_path = USERS_FILE.with_suffix(".tmp")
            tmp_path.write_text(
                yaml.safe_dump({"credentials": on_disk}), encoding="utf-8"
            )
            tmp_path.replace(USERS_FILE)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    logger.info("Credential store updated: password changed for %s", username)


def _save_temp_password_hash(username: str, hashed_password: str) -> None:
    """Atomically persist a caller-supplied bcrypt hash for username.

    Used by the custom forgot-password flow, which generates and hashes
    the temp password itself — it never goes through stauth's in-memory
    store, so this helper writes directly rather than reading from session.
    The cookie key and all env-sourced secrets are never written to disk.

    Also sets must_change_password: true so the user is forced to change
    their password on next login before accessing any other page.
    """
    lock_path = USERS_FILE.with_suffix(".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            on_disk = _load_credentials()
            if username not in on_disk["usernames"]:
                raise RuntimeError(
                    f"Cannot reset password: user {username!r} not found in {USERS_FILE}"
                )
            on_disk["usernames"][username]["password"] = hashed_password
            on_disk["usernames"][username]["must_change_password"] = True
            tmp_path = USERS_FILE.with_suffix(".tmp")
            tmp_path.write_text(
                yaml.safe_dump({"credentials": on_disk}), encoding="utf-8"
            )
            tmp_path.replace(USERS_FILE)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    logger.info(
        "Credential store updated: temporary password set for %s "
        "(must_change_password=True)",
        username,
    )


def get_must_change_password(username: str) -> bool:
    """Return True if the user must change their password before accessing the app.

    Reads fresh from users.yaml on every call so the flag is always current
    regardless of in-session credential cache state.
    """
    try:
        creds = _load_credentials()
        record = creds["usernames"].get(username) or {}
        return bool(record.get("must_change_password", False))
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to read must_change_password for %s: %s", username, exc
        )
        return False


def clear_must_change_password(username: str) -> None:
    """Atomically clear the must_change_password flag for username.

    Written under an exclusive lock + atomic rename so concurrent sessions
    cannot corrupt the credential store.
    """
    lock_path = USERS_FILE.with_suffix(".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            on_disk = _load_credentials()
            if username not in on_disk["usernames"]:
                raise RuntimeError(
                    f"Cannot clear flag: user {username!r} not found in {USERS_FILE}"
                )
            on_disk["usernames"][username].pop("must_change_password", None)
            tmp_path = USERS_FILE.with_suffix(".tmp")
            tmp_path.write_text(
                yaml.safe_dump({"credentials": on_disk}), encoding="utf-8"
            )
            tmp_path.replace(USERS_FILE)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    logger.info("Credential store updated: must_change_password cleared for %s", username)


def get_authenticator() -> stauth.Authenticate:
    """Return a per-session Authenticate instance backed by users.yaml.

    The _FacultyPasswordValidator is injected so that stauth's register and
    reset-password widgets enforce EduPilot's explicit password policy rather
    than the library's hardcoded defaults.
    """
    if _SS_AUTHENTICATOR not in st.session_state:
        st.session_state[_SS_AUTHENTICATOR] = stauth.Authenticate(
            _load_credentials(),
            cookie_name="edupilot_auth",
            cookie_key=_cookie_key(),
            cookie_expiry_days=7.0,
            validator=_FacultyPasswordValidator(),
        )
    return st.session_state[_SS_AUTHENTICATOR]


def try_cookie_login() -> None:
    """Attempt silent re-authentication from the session cookie (no UI)."""
    if st.session_state.get("authentication_status") is True:
        return
    try:
        get_authenticator().login(location="unrendered", key="CookieLogin")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cookie re-auth attempt failed: %s", exc)


def is_authenticated() -> bool:
    """True when the current session has a signed-in user with a valid username.

    Guards against a half-restored session (e.g. a cookie race) where
    ``authentication_status`` is True but ``username`` is empty/invalid —
    treating that as signed-out avoids crashing on every page render.
    """
    if st.session_state.get("authentication_status") is not True:
        return False
    raw = st.session_state.get("username") or ""
    if not _USERNAME_RE.match(raw):
        logger.warning(
            "Session marked authenticated but username is invalid (%r); "
            "treating as signed-out.", raw,
        )
        st.session_state["authentication_status"] = None
        return False
    return True


def current_username() -> str:
    """Validated username of the signed-in user (safe as a directory name).

    Raises:
        RuntimeError: If the session username fails the strict allowlist —
            we refuse to derive a filesystem path from anything else.
    """
    raw = st.session_state.get("username") or ""
    if not _USERNAME_RE.match(raw):
        raise RuntimeError(f"Invalid session username: {raw!r}")
    return raw


def current_display_name() -> str:
    """Human-readable name of the signed-in user."""
    return st.session_state.get("name") or st.session_state.get("username") or ""


def user_runs_dir() -> Path:
    """Per-user run-history directory (runs/<username>/). Created on demand."""
    username = current_username()
    base = config.runs_dir.resolve()
    d = (base / username).resolve()
    if d.parent != base:
        # Defense in depth — _USERNAME_RE already forbids path separators.
        raise RuntimeError(f"Refusing runs dir outside {base}: {d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_user_setting(key: str, default: str = "") -> str:
    """Read a per-user setting from runs/<username>/settings.json.

    Returns *default* on any error so callers never crash on a missing file.
    """
    try:
        settings_file = user_runs_dir() / "settings.json"
        if settings_file.exists():
            data = json.loads(settings_file.read_text(encoding="utf-8"))
            return str(data.get(key, default))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read user setting %r: %s", key, exc)
    return default


def save_user_setting(key: str, value: str) -> None:
    """Persist a per-user setting to runs/<username>/settings.json.

    Reads the existing file first so other keys are preserved.  Failures are
    logged but never propagated — a missing default is never fatal.
    """
    try:
        settings_file = user_runs_dir() / "settings.json"
        data: dict = {}
        if settings_file.exists():
            try:
                data = json.loads(settings_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                data = {}
        data[key] = value
        settings_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save user setting %r: %s", key, exc)


def _validate_new_username(username: str) -> bool:
    """Pre-registration check that the username is directory-safe."""
    return bool(_USERNAME_RE.match(username or ""))


def render_forgot_password() -> None:
    """Render the secure forgot-password recovery form.

    Security properties
    -------------------
    - Requires BOTH username AND the exact registered email (case-insensitive).
    - Returns one generic error for every failure mode — no enumeration.
    - Rate-limited to _FP_MAX_ATTEMPTS per _FP_WINDOW_SECS per session.
    - Temp password generated with secrets.token_urlsafe and hashed with
      bcrypt locally; persisted via _save_temp_password_hash (locked atomic
      write). stauth's forgot_password widget is NOT used — library never
      touches users.yaml.
    - Cookie key is never written to disk.
    - No st.rerun() called anywhere in this path.
    """
    sid = _fp_session_id()

    # --- Gate: check rate limit before showing the form --------------------
    if _fp_attempts_in_window(sid) >= _FP_MAX_ATTEMPTS:
        st.error(
            "Too many reset attempts. Please wait 15 minutes before trying again.",
            icon="🚫",
        )
        return

    st.caption(
        "Enter your username and registered email address. "
        "If both match our records, a temporary password will be shown here. "
        "Use it to sign in, then set a permanent password via **Change Password**."
    )

    with st.form("ForgotPasswordForm", clear_on_submit=True):
        username_input = st.text_input("Username")
        email_input = st.text_input("Registered email address")
        submitted = st.form_submit_button("Reset Password")

    if not submitted:
        return

    # --- Record the attempt immediately (before any validation) so that
    #     a fast retry loop cannot race through multiple checks. -----------
    _fp_record_attempt(sid)

    # Re-check after recording — handles the transition to the limit.
    if _fp_attempts_in_window(sid) > _FP_MAX_ATTEMPTS:
        st.error(
            "Too many reset attempts. Please wait 15 minutes before trying again.",
            icon="🚫",
        )
        return

    username = (username_input or "").strip()
    email_supplied = (email_input or "").strip().lower()

    # Reject obviously empty inputs with the SAME generic message.
    if not username or not email_supplied:
        st.error(_GENERIC_RESET_ERROR, icon="❌")
        return

    # --- Load credentials fresh from disk for verification ----------------
    try:
        creds = _load_credentials()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load credentials for password reset: %s", exc)
        st.error("A server error occurred. Please try again later.", icon="❌")
        return

    user_record = creds["usernames"].get(username)

    # Retrieve stored email whether or not the user exists, so the
    # comparison always runs (avoids a timing oracle on user existence).
    stored_email = (user_record or {}).get("email") or ""
    stored_email = stored_email.strip().lower()

    user_exists = user_record is not None
    email_matches = stored_email == email_supplied

    if not user_exists or not email_matches:
        # Identical path for every failure — no enumeration signal.
        logger.warning(
            "Forgot-password failed: username=%r exists=%s email_match=%s",
            username, user_exists, email_matches,
        )
        st.error(_GENERIC_RESET_ERROR, icon="❌")
        return

    # --- Both verified — generate and persist a temporary password --------
    tmp_password = _secrets.token_urlsafe(16)
    hashed = bcrypt.hashpw(tmp_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        _save_temp_password_hash(username, hashed)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist temp password for %s: %s", username, exc)
        st.error("A server error occurred. Please try again later.", icon="❌")
        return

    # Also mirror into the in-session stauth credential store so that login
    # with the temp password works in the same server process without a
    # restart (stauth holds credentials in memory per session).
    try:
        in_mem = get_authenticator().authentication_controller.authentication_model.credentials
        in_mem["usernames"][username]["password"] = hashed
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not mirror temp hash into session credentials for %s: %s",
            username, exc,
        )

    st.success("Temporary password generated — save it now!", icon="🔑")
    st.code(tmp_password, language=None)
    st.info(
        f"Sign in as **`{username}`** with the password above, "
        "then go to **Change Password** to set a permanent one.",
        icon="ℹ️",
    )
    logger.info("Temporary password issued for user: %s", username)


def render_auth_page() -> None:
    """Render the login / registration screen (shown when signed out)."""
    authenticator = get_authenticator()

    left, mid, right = st.columns([1, 1.3, 1])
    with mid:
        st.markdown(
            """
            <div style="text-align:center; padding: 1.2rem 0 0.4rem 0;">
                <div style="font-size:3rem;">🎓</div>
                <div style="font-size:1.9rem; font-weight:700;">EduPilot</div>
                <div style="opacity:0.65;">AI Faculty Assistant — sign in to continue</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register, tab_forgot = st.tabs(
            ["🔑 Sign in", "🆕 Create account", "❓ Forgot password"]
        )

        with tab_login:
            try:
                authenticator.login(location="main")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Login failed: {exc}", icon="❌")
                logger.error("Login widget error: %s", exc)

            status = st.session_state.get("authentication_status")
            if status is False:
                st.error("Incorrect username or password.", icon="❌")
            elif status is None:
                st.info("Enter your credentials, or create an account.", icon="👋")
            if status is True:
                # IMPORTANT: do NOT st.rerun() here — the auth cookie is
                # written by a frontend component that must finish mounting
                # on this run, or "stay signed in" silently breaks. The
                # cookie component triggers a rerun itself once it reports;
                # the button is a manual fallback.
                st.success("Signed in! Loading your workspace…", icon="✅")
                if st.button("Enter EduPilot →", type="primary"):
                    st.rerun()

        with tab_register:
            st.caption(
                "Username: 2–64 characters, letters, digits, `_` or `-` only."
            )
            st.caption(f"🔒 **Password policy:** {PASSWORD_POLICY_SUMMARY}")
            try:
                email, username, name = authenticator.register_user(
                    location="main",
                    captcha=False,
                    password_hint=False,
                    clear_on_submit=True,
                )
                if username:
                    # Retrieve what stauth stored in its in-memory model.
                    _reg_creds = authenticator.authentication_controller.authentication_model.credentials
                    _stored_email = (_reg_creds["usernames"].get(username) or {}).get("email") or ""
                    if not _validate_new_username(username):
                        st.error(
                            "Username may only contain letters, digits, "
                            "underscores and hyphens (2–64 characters).",
                            icon="❌",
                        )
                        # Remove the just-registered invalid user from the
                        # in-session store so it is never persisted.
                        _reg_creds["usernames"].pop(username, None)
                    elif not _stored_email.strip():
                        st.error(
                            "A valid email address is required for account recovery. "
                            "Please register again and provide your email.",
                            icon="❌",
                        )
                        # Remove the email-less record from in-session store.
                        _reg_creds["usernames"].pop(username, None)
                    else:
                        _save_new_user(username)
                        st.success(
                            f"Account created for **{name}** (`{username}`). "
                            "You can now sign in.",
                            icon="✅",
                        )
                        logger.info("New user registered: %s", username)
            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)
                if any(kw in exc_str.lower() for kw in ("password", "invalid", "weak", "must", "length")):
                    st.error(
                        f"❌ Password does not meet requirements. {PASSWORD_POLICY_SUMMARY}",
                        icon="❌",
                    )
                else:
                    st.error(f"Registration failed: {exc}", icon="❌")

        with tab_forgot:
            render_forgot_password()


def render_change_password(forced: bool = False) -> None:
    """Render the change-password widget for the currently signed-in user.

    Uses streamlit-authenticator's reset_password form (which validates the
    current password and enforces the library's password rules), then
    intercepts before stauth can write yaml: the updated bcrypt hash is
    persisted via the same locked, atomic save as new-user registration so
    the cookie key is never written to disk.

    Args:
        forced: When True the user arrived here via the must_change_password
                gate (i.e. they are using a temporary password).  On success
                the must_change_password flag is cleared so they can access
                the rest of the app on the next rerun.
    """
    username = current_username()

    st.subheader("🔐 Change Password")
    if forced:
        st.info(
            "Your account has a **temporary password**.  "
            "Please set a permanent password to continue — "
            "you cannot access any other page until this is done.",
            icon="🔒",
        )
    st.markdown(
        "Enter your current password, choose a new one, and confirm it."
    )
    st.caption(f"🔒 **Password policy:** {PASSWORD_POLICY_SUMMARY}")

    try:
        authenticator = get_authenticator()
        reset_ok = authenticator.reset_password(
            username=username,
            location="main",
            clear_on_submit=True,
            key="ChangePasswordForm",
        )
    except Exception as exc:  # noqa: BLE001
        # stauth raises on policy/validation failures — translate to a clear message.
        exc_str = str(exc)
        if any(kw in exc_str.lower() for kw in ("password", "invalid", "weak", "must", "length")):
            st.error(
                f"❌ New password does not meet the requirements. {PASSWORD_POLICY_SUMMARY}",
                icon="❌",
            )
        else:
            st.error(f"Password change failed: {exc}", icon="❌")
        logger.warning("reset_password widget error for %s: %s", username, exc)
        return

    if reset_ok is True:
        try:
            _save_updated_password(username)
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"Password updated in session but could not be saved to disk: {exc}",
                icon="❌",
            )
            logger.error("Failed to persist password change for %s: %s", username, exc)
            return

        # Clear the forced-change flag if this was a temp-password login.
        if forced:
            try:
                clear_must_change_password(username)
            except Exception as exc:  # noqa: BLE001
                # Non-fatal: password is already saved; log and let the user in.
                logger.error(
                    "Failed to clear must_change_password flag for %s: %s",
                    username, exc,
                )

        st.success(
            "✅ Password changed successfully. "
            "Your new password is active immediately.",
            icon="✅",
        )
        logger.info("Password changed for user: %s (forced=%s)", username, forced)
        if forced:
            # Allow the next rerun (triggered by user or the success message)
            # to pass the gate — do NOT st.rerun() here; the button below
            # lets the user trigger it manually so the widget unmounts cleanly.
            if st.button("Continue to EduPilot →", type="primary"):
                st.rerun()


def render_logout(location: str = "sidebar") -> None:
    """Render the logout button for the signed-in user.

    When the button is clicked, stauth clears the session *mid-render*,
    which would crash any later code that expects a signed-in user (e.g.
    per-user history reads).  So immediately after a successful logout we
    rerun the script — the auth gate then shows the login page cleanly.
    """
    try:
        get_authenticator().logout(button_name="🚪 Sign out", location=location)
    except KeyError:
        # stauth can raise KeyError('edupilot_auth') if the cookie is
        # already gone; treat it as a completed logout.
        for _k in ("authentication_status", "username", "name"):
            st.session_state[_k] = None
    if st.session_state.get("authentication_status") is not True:
        st.rerun()
