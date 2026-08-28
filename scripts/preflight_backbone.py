from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.config import load_config
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one bake-off backbone end to end.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.config.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(args.config)
    model = build_model(config)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total_parameters >= 2_000_000_000:
        raise RuntimeError(f"Parameter ceiling exceeded: {total_parameters:,}")

    device = torch.device("cuda")
    model.to(device).train()
    image_size = int(config["model"].get("image_size", 384))
    image = Image.new("RGB", (image_size + 37, image_size - 19), color=(91, 137, 203))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model([image])
        loss = output.aigc_logit.mean() + output.tamper_logit.mean()
    loss.backward()
    if model.token_adapter[1].weight.grad is None:
        raise RuntimeError("Adapter backward pass produced no gradient")

    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        state = {
            "token_adapter": model.token_adapter.state_dict(),
            "heads": model.heads.state_dict(),
        }
        torch.save(state, handle.name)
        restored = torch.load(handle.name, map_location="cpu", weights_only=True)
        model.token_adapter.load_state_dict(restored["token_adapter"])
        model.heads.load_state_dict(restored["heads"])

    result = {
        "backbone_type": config["model"].get("backbone_type", "gemma4"),
        "backbone_parameters": backbone_parameters,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "token_count": int(output.token_tamper_logits[0].numel()),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "forward_backward": "ok",
        "checkpoint_roundtrip": "ok",
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
