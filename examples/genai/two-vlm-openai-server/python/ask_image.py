#!/usr/bin/env python3
"""Send one local image prompt to a served VLM."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from urllib import error, request

import cv2

from utils.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True, help="Served model name, for example vlm-1")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        cfg.require_model(args.model)
        image = image_data_uri(args.image)
        response = request_vlm(cfg, args.model, image, args.prompt)
    except Exception as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 2

    print(response)
    return 0


def image_data_uri(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def request_vlm(cfg, model_name: str, image_uri: str, prompt: str) -> str:
    body = {
        "model": model_name,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }
    endpoint = f"http://{cfg.host}:{cfg.port}/v1/chat/completions"
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {exc.reason}: {detail}") from exc
    return parsed["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    raise SystemExit(main())
