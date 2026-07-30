"""Backwards-compatible entrypoint.

The service now lives in the :mod:`airavata_quant` package. This module keeps
``python airavata_quantization_service.py`` and the documented gunicorn target
``airavata_quantization_service:app`` working.
"""

from __future__ import annotations

import sys

from airavata_quant.api import app, create_app  # noqa: F401  (re-exported)
from airavata_quant.cli import main

__all__ = ["app", "create_app", "main"]


if __name__ == "__main__":
    argv = sys.argv[1:]
    # Bare invocation means "serve", matching the historical behaviour.
    if not argv or argv[0].startswith("-"):
        argv = ["serve", *argv]
    raise SystemExit(main(argv))
