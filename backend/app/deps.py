import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings
from app.model_service import ModelService


def get_service(request: Request) -> ModelService:
    return request.app.state.model


def require_api_key(
    x_api_key: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key header")
