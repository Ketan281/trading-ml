"""
Auth + user store — multi-user login/signup for Trading-AI.

DESIGN: deliberately small and dependency-light, in the spirit of the rest of
the system.
  • Users live in a SQLite DB (data/users.db) — same store the per-user wallet
    uses (aos/user_wallet.py), so a user and their book share one file.
  • Passwords are hashed with stdlib hashlib.pbkdf2_hmac (no bcrypt dependency),
    each with its own random salt.
  • Sessions are stateless JWT bearer tokens (PyJWT), 7-day expiry, signed with
    JWT_SECRET (env; random per-process fallback for dev).
  • Google OAuth: users can sign up / log in with a Google ID token. The token
    is verified against Google's public keys (google-auth library). A verified
    Google user is upserted with auth_provider='google' and no password hash.
  • Admin is the OWNER only: a signup whose email equals ADMIN_EMAIL
    (default ketanmohite8307@gmail.com) is granted role="admin". Everyone else
    is role="user".

This module exposes both the low-level user ops (signup/login) and the FastAPI
dependencies (current_user / admin_only) that guard the protected routes.
"""

import os
import sys
import hmac
import time
import sqlite3
import hashlib
import secrets
import logging
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jwt  # PyJWT
from fastapi import Depends, HTTPException, Header

DB = os.path.join(ROOT, "data", "users.db")
os.makedirs(os.path.dirname(DB), exist_ok=True)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "ketanmohite8307@gmail.com").lower().strip()
JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_hex(32)
JWT_ALGO = "HS256"
TOKEN_TTL = 7 * 24 * 3600          # 7 days
PBKDF2_ROUNDS = 200_000

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
_auth_log = logging.getLogger("auth")


# ── DB ────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create tables if absent. Safe to call repeatedly (and on import)."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                pwd_hash      TEXT NOT NULL,
                pwd_salt      TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                created_at    TEXT NOT NULL
            )""")
        cols = [r[1] for r in c.execute("PRAGMA table_info(users)")]
        if "auth_provider" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'email'")
        if "display_name" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
        if "avatar_url" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")


# ── password hashing ──────────────────────────────────
def _hash_pw(password, salt=None):
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ROUNDS)
    return dk.hex(), salt


def _verify_pw(password, pwd_hash, pwd_salt):
    calc, _ = _hash_pw(password, pwd_salt)
    return hmac.compare_digest(calc, pwd_hash)


# ── tokens ────────────────────────────────────────────
def make_token(user):
    now = int(time.time())
    payload = {"sub": str(user["id"]), "email": user["email"],
               "role": user["role"], "iat": now, "exp": now + TOKEN_TTL}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.PyJWTError:
        return None


# ── user ops ──────────────────────────────────────────
def _public(row):
    d = {"id": row["id"], "email": row["email"], "role": row["role"],
         "created_at": row["created_at"]}
    try:
        d["auth_provider"] = row["auth_provider"] or "email"
        d["display_name"] = row["display_name"]
        d["avatar_url"] = row["avatar_url"]
    except (IndexError, KeyError):
        d["auth_provider"] = "email"
    return d


def signup(email, password):
    email = (email or "").lower().strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError("a valid email is required")
    if not password or len(password) < 6:
        raise ValueError("password must be at least 6 characters")
    init_db()
    role = "admin" if email == ADMIN_EMAIL else "user"
    pwd_hash, salt = _hash_pw(password)
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO users (email, pwd_hash, pwd_salt, role, created_at) "
                "VALUES (?,?,?,?,?)",
                (email, pwd_hash, salt, role, datetime.now(timezone.utc).isoformat()))
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("an account with that email already exists")
    user = {"id": uid, "email": email, "role": role,
            "created_at": datetime.now(timezone.utc).isoformat()}
    return {"token": make_token(user), "user": user}


def login(email, password):
    email = (email or "").lower().strip()
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not _verify_pw(password, row["pwd_hash"], row["pwd_salt"]):
        raise ValueError("invalid email or password")
    user = _public(row)
    return {"token": make_token(user), "user": user}


def get_user(uid):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (int(uid),)).fetchone()
    return _public(row) if row else None


def list_users():
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [_public(r) for r in rows]


def change_password(uid, old_password, new_password):
    """Verify the current password, then set a new one (>= 6 chars)."""
    if not new_password or len(new_password) < 6:
        raise ValueError("new password must be at least 6 characters")
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (int(uid),)).fetchone()
        if not row:
            raise ValueError("user not found")
        if not _verify_pw(old_password, row["pwd_hash"], row["pwd_salt"]):
            raise ValueError("current password is incorrect")
        pwd_hash, salt = _hash_pw(new_password)
        c.execute("UPDATE users SET pwd_hash=?, pwd_salt=? WHERE id=?",
                  (pwd_hash, salt, int(uid)))
    return {"ok": True}


def set_role(uid, role):
    """Admin op: promote/demote a user. Returns the updated public user."""
    role = (role or "").lower().strip()
    if role not in ("user", "admin"):
        raise ValueError("role must be 'user' or 'admin'")
    with _conn() as c:
        cur = c.execute("UPDATE users SET role=? WHERE id=?", (role, int(uid)))
        if cur.rowcount == 0:
            raise ValueError("user not found")
    return get_user(uid)


# ── Google OAuth ─────────────────────────────────────
def _verify_google_token(id_token_str):
    """Verify a Google ID token and return the claims (email, name, picture).
    Uses google-auth if available, falls back to manual JWT decode against
    Google's public keys."""
    try:
        from google.oauth2 import id_token as g_id_token
        from google.auth.transport import requests as g_requests
        claims = g_id_token.verify_oauth2_token(
            id_token_str, g_requests.Request(), GOOGLE_CLIENT_ID)
        return claims
    except ImportError:
        pass
    # Fallback: decode the JWT without google-auth (for dev environments).
    # Google tokens are RS256-signed; we verify the issuer + audience manually
    # using PyJWT + Google's JWKS.
    try:
        import urllib.request, json as _json
        jwks_url = "https://www.googleapis.com/oauth2/v3/certs"
        jwks_data = _json.loads(urllib.request.urlopen(jwks_url, timeout=5).read())
        from jwt import PyJWKSet
        keyset = PyJWKSet.from_dict(jwks_data)
        header = jwt.get_unverified_header(id_token_str)
        key = None
        for k in keyset.keys:
            if k.key_id == header.get("kid"):
                key = k; break
        if not key:
            raise ValueError("no matching Google signing key")
        claims = jwt.decode(id_token_str, key.key,
                            algorithms=["RS256"],
                            audience=GOOGLE_CLIENT_ID,
                            issuer=["accounts.google.com", "https://accounts.google.com"])
        return claims
    except Exception as e:
        raise ValueError(f"Google token verification failed: {e}")


def google_auth(id_token_str):
    """Sign up or log in a user via Google OAuth. Returns {token, user}."""
    if not GOOGLE_CLIENT_ID:
        raise ValueError("Google OAuth is not configured — set GOOGLE_CLIENT_ID env var")
    claims = _verify_google_token(id_token_str)
    email = claims.get("email", "").lower().strip()
    if not email or not claims.get("email_verified"):
        raise ValueError("Google account email is not verified")
    display_name = claims.get("name", "")
    avatar_url = claims.get("picture", "")
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if row:
        try:
            with _conn() as c:
                c.execute("UPDATE users SET display_name=?, avatar_url=?, auth_provider='google' "
                          "WHERE email=?", (display_name, avatar_url, email))
        except Exception:
            pass
        user = _public(row)
        user["display_name"] = display_name
        user["avatar_url"] = avatar_url
        return {"token": make_token(user), "user": user}
    role = "admin" if email == ADMIN_EMAIL else "user"
    sentinel_hash, sentinel_salt = _hash_pw(secrets.token_hex(32))
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO users (email, pwd_hash, pwd_salt, role, created_at, "
            "auth_provider, display_name, avatar_url) VALUES (?,?,?,?,?,?,?,?)",
            (email, sentinel_hash, sentinel_salt, role, now,
             "google", display_name, avatar_url))
        uid = cur.lastrowid
    user = {"id": uid, "email": email, "role": role, "created_at": now,
            "auth_provider": "google", "display_name": display_name,
            "avatar_url": avatar_url}
    _auth_log.info("Google signup: %s", email)
    return {"token": make_token(user), "user": user}


# ── FastAPI dependencies ──────────────────────────────
def current_user(authorization: str = Header(None)):
    """Resolve the bearer token to a user dict, or 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    payload = decode_token(authorization.split(" ", 1)[1].strip())
    if not payload:
        raise HTTPException(401, "invalid or expired token")
    user = get_user(payload["sub"])
    if not user:
        raise HTTPException(401, "user no longer exists")
    return user


def admin_only(user: dict = Depends(current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "admin access only")
    return user


init_db()


if __name__ == "__main__":
    # Quick self-test against a throwaway DB.
    import tempfile
    DB = os.path.join(tempfile.gettempdir(), "trading_ai_auth_test.db")
    if os.path.exists(DB):
        os.remove(DB)
    init_db()
    print("signup owner:", signup(ADMIN_EMAIL, "secret123")["user"])
    print("signup user :", signup("alice@example.com", "secret123")["user"])
    try:
        signup("alice@example.com", "secret123")
    except ValueError as e:
        print("dup rejected:", e)
    tok = login("alice@example.com", "secret123")["token"]
    print("login token len:", len(tok), "| decoded:", decode_token(tok)["email"])
    try:
        login("alice@example.com", "wrong")
    except ValueError as e:
        print("bad pw rejected:", e)
    print("users:", [u["email"] + "/" + u["role"] for u in list_users()])
    try:
        os.remove(DB)
    except OSError:
        pass  # Windows may still hold the sqlite handle; harmless for a test DB
