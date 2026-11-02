import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.model_service import ModelService
from app.routers import insights, predict

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("churnsense")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelService(settings.artifacts_dir)
    yield


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="Customer churn prediction, explanations and retention ROI simulation.",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error id=%s path=%s", request_id, request.url.path)
        response = JSONResponse({"detail": "Internal server error", "request_id": request_id}, status_code=500)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    log.info("%s %s %s %.1fms id=%s", request.method, request.url.path, response.status_code, elapsed, request_id)
    return response


@app.get("/", include_in_schema=False)
def root():
    return {"name": settings.app_name, "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["ops"])
def health(request: Request):
    model = request.app.state.model
    return {
        "status": "ok",
        "model_version": model.version,
        "model_name": model.model_name,
        "trained_at": model.trained_at,
        "environment": settings.environment,
    }


app.include_router(predict.router, prefix="/api/v1")
app.include_router(insights.router, prefix="/api/v1")
