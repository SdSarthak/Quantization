"""FastAPI application exposing generation, benchmarking and telemetry."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, hardware
from .config import Settings
from .manager import ModelLoadError, ModelManager, VariantUnavailableError
from .quantization import VARIANTS, UnknownVariantError, memory_ratio
from .schemas import (
    BenchmarkAllResponse,
    BenchmarkResponse,
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    ModelsResponse,
    SaveResponse,
    SystemInfoResponse,
    VariantInfo,
)

logger = logging.getLogger(__name__)


def get_manager(request: Request) -> ModelManager:
    """FastAPI dependency returning the app-scoped manager."""
    return request.app.state.manager


async def _run(manager: ModelManager, func, *args, **kwargs):
    """Execute a blocking manager call on the manager's thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        manager.executor, lambda: func(*args, **kwargs)
    )


def _translate(exc: Exception) -> HTTPException:
    """Map domain errors onto HTTP status codes."""
    if isinstance(exc, UnknownVariantError):
        return HTTPException(
            status_code=404,
            detail=f"unknown model type '{exc.name}'. Available: {sorted(VARIANTS)}",
        )
    if isinstance(exc, VariantUnavailableError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ModelLoadError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("unhandled error while serving request")
    return HTTPException(status_code=500, detail=str(exc))


def create_app(
    settings: Optional[Settings] = None,
    manager: Optional[ModelManager] = None,
) -> FastAPI:
    """Build the application.

    Both the settings and the manager are injectable so tests can drive the
    routes against a stub instead of a multi-gigabyte checkpoint.
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        if app.state.manager is None:
            app.state.manager = ModelManager(settings)
        app.state.preload_errors = {}

        try:
            settings.ensure_directories()
            errors = await asyncio.get_running_loop().run_in_executor(
                None, app.state.manager.preload
            )
            app.state.preload_errors = errors
        except Exception as exc:  # noqa: BLE001 - never block startup
            # Serving /health and /system/info is still valuable when weights
            # are unavailable, and lazy loading may succeed later.
            logger.error("startup preload failed: %s", exc)
            app.state.preload_errors = {"startup": str(exc)}

        yield

        app.state.manager.shutdown()

    app = FastAPI(
        title="Airavata Quantization Service",
        description=(
            "Serve the AI4Bharat Airavata model under several quantization "
            "schemes and benchmark them against each other."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.settings = settings
    app.state.preload_errors = {}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    @app.get("/")
    async def root(manager: ModelManager = Depends(get_manager)) -> Dict[str, Any]:
        return {
            "service": "Airavata Quantization Service",
            "version": __version__,
            "model_name": manager.settings.model_name,
            "device": str(manager.device),
            "loaded_models": sorted(manager.models),
            "available_models": manager.available_variants(),
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health(manager: ModelManager = Depends(get_manager)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            device=str(manager.device),
            model_name=manager.settings.model_name,
            loaded_models=sorted(manager.models),
            tokenizer_ready=manager.tokenizer is not None,
        )

    @app.get("/models", response_model=ModelsResponse)
    async def models(manager: ModelManager = Depends(get_manager)) -> ModelsResponse:
        return ModelsResponse(
            device=str(manager.device),
            model_name=manager.settings.model_name,
            loaded=sorted(manager.models),
            variants=[
                VariantInfo(
                    name=variant.name,
                    description=variant.description,
                    bits=variant.bits,
                    requires_device=variant.requires_device,
                    supported=manager.supports(variant.name),
                    loaded=variant.name in manager.models,
                    relative_weight_memory=memory_ratio(
                        variant.name, manager.device.type
                    ),
                )
                for variant in VARIANTS.values()
            ],
        )

    @app.post("/generate/{model_type}", response_model=GenerationResponse)
    async def generate(
        model_type: str,
        request: GenerationRequest,
        manager: ModelManager = Depends(get_manager),
    ) -> GenerationResponse:
        try:
            result = await _run(manager, manager.generate, model_type, request)
        except Exception as exc:  # noqa: BLE001 - translated below
            raise _translate(exc) from exc
        return GenerationResponse(**result)

    # NOTE: /benchmark/all must be declared before /benchmark/{model_type}.
    # FastAPI matches in declaration order, so the parameterised route would
    # otherwise swallow "all" and 404.
    @app.get("/benchmark/all", response_model=BenchmarkAllResponse)
    async def benchmark_all(
        iterations: int = Query(3, ge=1, le=100),
        max_new_tokens: int = Query(50, ge=1, le=2048),
        keep_loaded: bool = Query(
            False,
            description=(
                "Keep every benchmarked variant resident. Off by default so "
                "the sweep fits on a device that holds one copy of the weights."
            ),
        ),
        manager: ModelManager = Depends(get_manager),
    ) -> BenchmarkAllResponse:
        sweep = await _run(
            manager,
            manager.benchmark_all,
            None,
            iterations,
            None,
            max_new_tokens,
            not keep_loaded,
        )
        return BenchmarkAllResponse(
            iterations=iterations,
            results={
                name: BenchmarkResponse(**stats)
                for name, stats in sweep["results"].items()
            },
            comparison=sweep["comparison"],
            errors=sweep["errors"],
        )

    @app.get("/benchmark/{model_type}", response_model=BenchmarkResponse)
    async def benchmark(
        model_type: str,
        iterations: int = Query(5, ge=1, le=100),
        max_new_tokens: int = Query(50, ge=1, le=2048),
        manager: ModelManager = Depends(get_manager),
    ) -> BenchmarkResponse:
        try:
            stats = await _run(
                manager, manager.benchmark, model_type, iterations, None, max_new_tokens
            )
        except Exception as exc:  # noqa: BLE001 - translated below
            raise _translate(exc) from exc
        return BenchmarkResponse(**stats)

    @app.post("/models/{model_type}/save", response_model=SaveResponse)
    async def save(
        model_type: str, manager: ModelManager = Depends(get_manager)
    ) -> SaveResponse:
        try:
            result = await _run(manager, manager.save, model_type)
        except Exception as exc:  # noqa: BLE001 - translated below
            raise _translate(exc) from exc
        return SaveResponse(**result)

    @app.delete("/models/{model_type}")
    async def unload(
        model_type: str, manager: ModelManager = Depends(get_manager)
    ) -> Dict[str, Any]:
        removed = manager.unload(model_type)
        if not removed:
            raise HTTPException(
                status_code=404, detail=f"model type '{model_type}' is not loaded"
            )
        return {"unloaded": model_type, "loaded_models": sorted(manager.models)}

    @app.get("/system/info", response_model=SystemInfoResponse)
    async def system_info(
        manager: ModelManager = Depends(get_manager),
    ) -> SystemInfoResponse:
        return SystemInfoResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            device=str(manager.device),
            model_name=manager.settings.model_name,
            loaded_models=sorted(manager.models),
            cpu=hardware.cpu_info(),
            memory=hardware.memory_info(),
            gpu=hardware.gpu_info(),
            gpu_name=hardware.gpu_name(),
        )

    return app


app = create_app()
