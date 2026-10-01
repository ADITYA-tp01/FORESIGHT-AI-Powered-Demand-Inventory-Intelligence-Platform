"""FastAPI scoring service (docs/Plan.md 8).

Run from the project root: ``uvicorn api.main:app`` (or ``python -m uvicorn api.main:app``).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from foresight import __version__

from .routes import router
from .store import ServingUnavailableError, UnknownSkuError

app = FastAPI(title="FORESIGHT Scoring API", version=__version__)
app.include_router(router)


@app.exception_handler(UnknownSkuError)
def unknown_sku_handler(request: Request, exc: UnknownSkuError) -> JSONResponse:
    if len(exc.sku_ids) == 1:
        payload = {"error": "unknown_sku", "sku_id": exc.sku_ids[0]}
    else:
        payload = {"error": "unknown_sku_ids", "missing": exc.sku_ids}
    return JSONResponse(status_code=404, content=payload)


@app.exception_handler(ServingUnavailableError)
def serving_unavailable_handler(request: Request, exc: ServingUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": "serving_unavailable", "detail": str(exc)},
    )
