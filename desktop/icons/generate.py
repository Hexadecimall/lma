"""Rebuild the dependency-free RGBA app icon and Windows ICO."""

import math
import struct
import zlib
from pathlib import Path

SIZE = 256
ROOT = Path(__file__).resolve().parent


def segment_distance(x, y, start, end):
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    t = max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - ax - t * dx, y - ay - t * dy)


def color(x, y):
    # Rounded charcoal tile with a sage lambda monogram.
    if math.hypot(max(36 - x, 0, x - 220), max(36 - y, 0, y - 220)) > 32:
        return (0, 0, 0, 0)
    strokes = [((92, 69), (111, 69)), ((111, 69), (174, 188)), ((129, 105), (77, 188))]
    if min(segment_distance(x, y, a, b) for a, b in strokes) < 10:
        return (182, 207, 188, 255)
    return (25, 26, 28, 255)


def chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


rows = bytearray()
for y in range(SIZE):
    rows.append(0)
    for x in range(SIZE):
        samples = [
            color(x + dx, y + dy)
            for dx, dy in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        ]
        rows.extend(
            round(sum(sample[channel] for sample in samples) / 4)
            for channel in range(4)
        )
png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows, 9))
    + chunk(b"IEND", b"")
)
(ROOT / "icon.png").write_bytes(png)
# Windows Vista+ accepts PNG-compressed 256px images in an ICO directory.
ico = (
    struct.pack("<HHH", 0, 1, 1)
    + struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 22)
    + png
)
(ROOT / "icon.ico").write_bytes(ico)
print("Generated icon.png and icon.ico")
