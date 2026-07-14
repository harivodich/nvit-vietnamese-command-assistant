"""FastAPI adapter mỏng cho pipeline; model được nạp một lần trong lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request

from nvit_assistant.nlu.pipeline import NLUPipeline
from nvit_assistant.runtime import build_pipeline
from nvit_assistant.schemas import ParseRequest, ParseResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(pipeline: NLUPipeline | None = None) -> FastAPI:
    """Tạo app có thể inject pipeline trong test và lazy-load pipeline ở production."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            app.state.pipeline = build_pipeline(PROJECT_ROOT)
        yield

    application = FastAPI(
        title="NVIT Vietnamese Command Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request) -> dict[str, str]:
        """Báo ready chỉ sau khi lifespan đã nạp pipeline thành công."""
        status = "ready" if hasattr(request.app.state, "pipeline") else "starting"
        return {"status": status, "mode": "mock-actions"}

    @application.post("/parse", response_model=ParseResult)
    def parse(payload: ParseRequest, request: Request) -> ParseResult:
        """Validate request bằng Pydantic rồi chuyển thẳng vào core pipeline dùng chung."""
        runtime_pipeline: NLUPipeline = request.app.state.pipeline
        return runtime_pipeline.parse(payload)

    return application


app = create_app()
