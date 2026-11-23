"""Tiny HTTP gateway in front of the torchcast server.

Why: the deployed web_server answered GET /health instantly but hung on POST
/v1/systemone. Running the same POST against the server on loopback inside the
container returned in 0.10 s, so the model is fine and the hang lives in the
proxy path to the bare-socket server. This gateway takes a normal HTTP request
(Content-Length set, connection closed by the client library) and forwards it to
127.0.0.1:8011, which the server handles correctly.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

UPSTREAM = os.environ.get("TC_UPSTREAM", "http://127.0.0.1:8011")
app = FastAPI(title="torchcast-gateway")


@app.get("/health")
def health() -> Response:
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=5) as resp:
            return Response(resp.read(), media_type="application/json")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"status": "starting", "error": str(exc)}, status_code=503)


@app.get("/v1/models")
def models() -> Response:
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/v1/models", timeout=5) as resp:
            return Response(resp.read(), media_type="application/json")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": {"code": 502, "message": str(exc)}}, status_code=502)


@app.post("/v1/systemone")
async def systemone(request: Request) -> Response:
    body = await request.body()
    req = urllib.request.Request(
        f"{UPSTREAM}/v1/systemone",
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(resp.read(), status_code=resp.status, media_type="application/json")
    except urllib.error.HTTPError as exc:
        return Response(exc.read(), status_code=exc.code, media_type="application/json")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": {"code": 502, "message": str(exc)}}, status_code=502)


@app.get("/")
def root() -> dict:
    return {"service": "torchcast-decision gateway", "upstream": UPSTREAM,
            "endpoints": ["/health", "/v1/models", "/v1/systemone"]}
