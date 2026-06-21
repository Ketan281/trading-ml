import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from api import auth
from aos import user_wallet_views as uw_views

router = APIRouter(tags=["live"])


def _resolve_stream_user(token: str):
    if not token:
        raise HTTPException(401, "missing token")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "invalid or expired token")
    user = auth.get_user(payload["sub"])
    if not user:
        raise HTTPException(401, "user not found")
    return user


def _resolve_sse_user(request: Request):
    return _resolve_stream_user(request.query_params.get("token"))


@router.get("/me/live")
async def me_live(request: Request):
    user = _resolve_sse_user(request)
    uid = user["id"]

    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            try:
                data = uw_views.live_snapshot(uid)
                yield f"data: {json.dumps(data, default=str)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/me/live/{trade_id}")
async def me_live_trade(trade_id: str, request: Request):
    user = _resolve_sse_user(request)
    uid = user["id"]

    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            try:
                data = uw_views.trade_snapshot(uid, trade_id)
                if data.get("error"):
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                yield f"data: {json.dumps(data, default=str)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    try:
        user = _resolve_stream_user(websocket.query_params.get("token"))
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    uid = user["id"]
    try:
        while True:
            try:
                await websocket.send_json(uw_views.live_snapshot(uid))
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
            await asyncio.sleep(2)
    except (WebSocketDisconnect, Exception):
        pass
