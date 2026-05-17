"""Shared image montage helpers for analysis exports."""

from __future__ import annotations

import math
from typing import List, Tuple

from PIL import Image, ImageDraw


RGBColor = Tuple[int, int, int]
BORDER_COLOR: RGBColor = (210, 210, 210)


def make_tile(
    img: Image.Image | None,
    tile_size: int,
    border: int,
    bg_color: RGBColor = (255, 255, 255),
    scale: bool = True,
) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size), bg_color)
    if img is None:
        draw = ImageDraw.Draw(canvas)
        for b in range(border):
            draw.rectangle([b, b, tile_size - 1 - b, tile_size - 1 - b], outline=BORDER_COLOR)
        return canvas

    w, h = img.size
    if scale:
        max_w = tile_size - 2 * border - 2
        max_h = tile_size - 2 * border - 2
        if max_w <= 0 or max_h <= 0:
            max_w = max_h = max(1, tile_size - 2)
        ratio = min(max_w / max(1, w), max_h / max(1, h))
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    else:
        img_resized = img
        new_w, new_h = img.size

    ox = (tile_size - new_w) // 2
    oy = (tile_size - new_h) // 2
    canvas.paste(img_resized.convert("RGB"), (ox, oy))

    draw = ImageDraw.Draw(canvas)
    for b in range(border):
        draw.rectangle([b, b, tile_size - 1 - b, tile_size - 1 - b], outline=BORDER_COLOR)
    return canvas


def build_montage(
    tiles: List[Image.Image],
    cols: int,
    tile_size: int,
    bg_color: RGBColor = (255, 255, 255),
) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (tile_size, tile_size), bg_color)

    cols = max(1, cols)
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (cols * tile_size, rows * tile_size), bg_color)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        out.paste(tile, (c * tile_size, r * tile_size))
    return out


def hex_to_rgb(hex_str: str) -> RGBColor:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Invalid hex color, expect #RRGGBB")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
