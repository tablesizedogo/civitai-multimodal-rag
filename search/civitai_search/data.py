"""Loading the metadata corpus."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import DATA_DIR, META_PATH


@dataclass
class Item:
    id: int
    path: Path
    page_url: str
    base_model: str
    prompt: str
    negative_prompt: str
    model: str | None
    sampler: str | None
    steps: int | None
    cfg_scale: float | None
    seed: int | None
    loras: list
    has_comfy_workflow: bool
    image_url: str = ""  # Civitai CDN (width=512) - used when the local file is absent (HF Space)

    def image_ref(self) -> str:
        """Local path if the image is on disk, otherwise the CDN URL."""
        return str(self.path) if self.path.exists() else (self.image_url or str(self.path))


def _row_to_item(r: dict) -> Item:
    return Item(
        id=int(r["id"]),
        path=DATA_DIR / r["local_path"],
        page_url=r.get("page_url") or f"https://civitai.com/images/{r['id']}",
        base_model=r.get("baseModel") or "",
        prompt=r.get("prompt") or "",
        negative_prompt=r.get("negativePrompt") or "",
        model=r.get("Model"),
        sampler=r.get("sampler"),
        steps=r.get("steps"),
        cfg_scale=r.get("cfgScale"),
        seed=r.get("seed"),
        loras=r.get("loras") or [],
        has_comfy_workflow=bool(r.get("has_comfy_workflow")),
        image_url=r.get("download_url") or r.get("image_url") or "",
    )


@lru_cache(maxsize=1)
def load_items(meta_path: Path = META_PATH) -> list[Item]:
    """All items in a fixed order (sorted by id). Embedding row i == items[i]."""
    with open(meta_path, encoding="utf-8") as f:
        items = [_row_to_item(json.loads(line)) for line in f if line.strip()]
    items.sort(key=lambda it: it.id)
    return items


def model_family(base_model: str) -> str:
    b = base_model.lower()
    if b.startswith("krea"):
        return "Krea 2"
    if b.startswith("flux"):
        return "Flux"
    if "illustrious" in b:
        return "Illustrious"
    if b.startswith("zimage") or b.startswith("z image"):
        return "Z Image"
    if b.startswith("sdxl"):
        return "SDXL"
    if b.startswith("pony"):
        return "Pony"
    return base_model or "unknown"
