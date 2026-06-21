from fastapi import APIRouter, Depends, HTTPException

from api import auth
from api.schemas import Credentials, GoogleToken, PasswordChange

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup")
def auth_signup(body: Credentials):
    try:
        return auth.signup(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/login")
def auth_login(body: Credentials):
    try:
        return auth.login(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(401, str(exc))


@router.post("/google")
def auth_google(body: GoogleToken):
    try:
        return auth.google_auth(body.id_token)
    except ValueError as exc:
        raise HTTPException(401, str(exc))


@router.get("/me")
def auth_me(user: dict = Depends(auth.current_user)):
    return user


@router.get("/config")
def auth_config():
    return {"google_client_id": auth.GOOGLE_CLIENT_ID or None}


@router.post("/change-password")
def auth_change_password(body: PasswordChange, user: dict = Depends(auth.current_user)):
    try:
        return auth.change_password(user["id"], body.old_password, body.new_password)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
