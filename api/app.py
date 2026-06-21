"""Canonical FastAPI app assembly."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.admin_routes import router as admin_router
from api.routers.auth_routes import router as auth_router
from api.routers.live_routes import router as live_router
from api.routers.public_routes import router as public_router
from api.routers.user_routes import router as user_router


def create_app():
    app = FastAPI(
        title="Trading-AI",
        version="1.0",
        description="NSE equity + NIFTY/BANKNIFTY options intelligence",
    )

    origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
               if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.include_router(public_router)
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(live_router)
    app.include_router(admin_router)

    return app


app = create_app()
