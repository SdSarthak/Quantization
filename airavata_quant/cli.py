"""Command line entrypoints: ``python -m airavata_quant <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, Settings


def _configure_logging(level: str) -> None:
    # "TRACE" is a uvicorn level with no stdlib equivalent; map it to DEBUG.
    resolved = logging.DEBUG if level.upper() == "TRACE" else level.upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO) if isinstance(resolved, str) else resolved,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _positive_int(raw: str) -> int:
    """argparse type that rejects 0/negatives up front instead of mid-benchmark."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airavata-quant",
        description="Serve, benchmark and export quantized Airavata models.",
    )
    parser.add_argument("--model", help="Override the Hugging Face model id.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Force a device instead of auto-detecting.",
    )
    parser.add_argument("--log-level", help="Logging level (default from env).")

    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP API.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=_positive_int)
    serve.add_argument(
        "--reload", action="store_true", help="Enable uvicorn autoreload."
    )
    serve.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip loading models at startup; load them on first request.",
    )

    bench = sub.add_parser("benchmark", help="Benchmark one or all variants.")
    bench.add_argument(
        "variants",
        nargs="*",
        help="Variant names to benchmark. Defaults to everything the device supports.",
    )
    bench.add_argument("--iterations", type=_positive_int, default=3)
    bench.add_argument("--max-new-tokens", type=_positive_int, default=50)
    bench.add_argument("--prompt", action="append", dest="prompts")
    bench.add_argument("--output", type=Path, help="Write the JSON report here.")
    bench.add_argument(
        "--keep-loaded",
        action="store_true",
        help=(
            "Keep every benchmarked variant in memory. By default each one is "
            "released after it is measured so the sweep fits on a device that "
            "holds a single copy of the weights."
        ),
    )

    export = sub.add_parser("export", help="Save a quantized variant to disk.")
    export.add_argument("variant")
    export.add_argument("--output", type=Path)

    sub.add_parser("info", help="Print device, config and variant availability.")

    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if args.model:
        settings.model_name = args.model
    if args.device:
        settings.device = args.device
    if args.log_level:
        settings.log_level = args.log_level.upper()
    if getattr(args, "host", None):
        settings.host = args.host
    if getattr(args, "port", None):
        settings.port = args.port
    if getattr(args, "no_preload", False):
        settings.preload = []
    # Assigning to dataclass fields skips __post_init__, so re-check the
    # overrides rather than letting a bad --port or --log-level reach uvicorn.
    return settings.validate()


def _cmd_info(settings: Settings) -> int:
    from .manager import ModelManager
    from .quantization import VARIANTS, memory_ratio

    manager = ModelManager(settings)
    payload = {
        "model_name": settings.model_name,
        "device": str(manager.device),
        "cache_dir": str(settings.cache_dir),
        "quantized_model_path": str(settings.quantized_model_path),
        "preload": settings.preload,
        "hf_token_configured": bool(settings.hf_token),
        "variants": {
            name: {
                "description": variant.description,
                "requires_device": variant.requires_device,
                "supported_here": manager.supports(name),
                "relative_weight_memory": memory_ratio(name, manager.device.type),
            }
            for name, variant in VARIANTS.items()
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_serve(settings: Settings, reload: bool) -> int:
    import uvicorn

    settings.ensure_directories()
    if reload:
        # Reload requires an import string; the app factory reads the same env.
        uvicorn.run(
            "airavata_quant.api:app",
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level=settings.log_level.lower(),
        )
        return 0

    from .api import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    return 0


def _cmd_benchmark(settings: Settings, args: argparse.Namespace) -> int:
    from .manager import ModelManager

    settings.ensure_directories()
    manager = ModelManager(settings)
    try:
        sweep = manager.benchmark_all(
            variants=args.variants or None,
            iterations=args.iterations,
            prompts=args.prompts,
            max_new_tokens=args.max_new_tokens,
            free_after=not args.keep_loaded,
        )
        for name, message in sweep["errors"].items():
            print(f"benchmark of {name} failed: {message}", file=sys.stderr)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": settings.model_name,
            "device": str(manager.device),
            "iterations": args.iterations,
            "max_new_tokens": args.max_new_tokens,
            **sweep,
        }

        output = args.output
        if output is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = settings.benchmark_dir / f"benchmark-{stamp}.json"
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            # A benchmark that ran is worth printing even if the report cannot
            # be filed; do not throw the measurements away over a bad --output.
            print(f"could not write the report to {output}: {exc}", file=sys.stderr)
        else:
            print(f"report written to {output}", file=sys.stderr)

        print(json.dumps(report, indent=2))
        return 0 if report["results"] else 1
    finally:
        manager.shutdown()


def _cmd_export(settings: Settings, args: argparse.Namespace) -> int:
    from .manager import ModelManager

    settings.ensure_directories()
    manager = ModelManager(settings)
    result = manager.save(args.variant, destination=args.output)
    print(json.dumps(result, indent=2))
    manager.shutdown()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        settings = _settings_from_args(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    _configure_logging(settings.log_level)

    try:
        if args.command == "serve":
            return _cmd_serve(settings, reload=args.reload)
        if args.command == "benchmark":
            return _cmd_benchmark(settings, args)
        if args.command == "export":
            return _cmd_export(settings, args)
        if args.command == "info":
            return _cmd_info(settings)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI reports instead of tracebacks
        logging.getLogger(__name__).debug("command failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
