"""Image grids for inspection and comparison."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE = 192


def _font(size: int = 14):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _thumb(path: Path, tile: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((tile, tile))
    canvas = Image.new("RGB", (tile, tile), (32, 32, 32))
    canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
    return canvas


def image_grid(paths, out: Path, cols: int = 8, labels=None, tile: int = TILE) -> Path:
    rows = (len(paths) + cols - 1) // cols
    g = Image.new("RGB", (cols * tile, rows * tile), (20, 20, 20))
    d = ImageDraw.Draw(g)
    f = _font(13)
    for k, p in enumerate(paths):
        x, y = (k % cols) * tile, (k // cols) * tile
        g.paste(_thumb(p, tile), (x, y))
        if labels:
            d.rectangle([x, y, x + 70, y + 18], fill=(0, 0, 0))
            d.text((x + 3, y + 2), str(labels[k]), fill=(255, 255, 0), font=f)
    g.save(out)
    return out


def comparison_grid(title: str, rows: list[tuple[str, list[Path]]], out: Path,
                    k: int = 10, tile: int = 160) -> Path:
    """One row per method, k tiles per row, numbered 1..k."""
    label_w, head_h = 170, 34
    g = Image.new("RGB", (label_w + k * tile, head_h + len(rows) * tile), (20, 20, 20))
    d = ImageDraw.Draw(g)
    d.text((8, 8), title, fill=(255, 255, 255), font=_font(18))
    fs = _font(12)
    for r, (name, paths) in enumerate(rows):
        y = head_h + r * tile
        d.text((8, y + tile // 2 - 8), name, fill=(200, 220, 255), font=_font(14))
        for c, p in enumerate(paths[:k]):
            x = label_w + c * tile
            g.paste(_thumb(p, tile), (x, y))
            d.rectangle([x, y, x + 20, y + 16], fill=(0, 0, 0))
            d.text((x + 3, y + 1), str(c + 1), fill=(255, 255, 0), font=fs)
    g.save(out)
    return out
