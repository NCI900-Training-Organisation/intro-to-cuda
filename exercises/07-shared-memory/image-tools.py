"""Dependency-free grayscale scene, CPU reference, PNG writer, and reduction diagram."""
import math
import struct
import zlib
from pathlib import Path


def make_scene(width=384, height=256, darkness=0.35):
    if not (1 <= width <= 4096 and 1 <= height <= 4096 and 0 < darkness <= 1):
        raise ValueError('Use dimensions 1..4096 and darkness in (0, 1]')
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            u, v = x / width, y / height
            value = 180 - 50 * v
            if (u - 0.76)**2 + (v - 0.22)**2 < 0.085**2:
                value = 245  # Sun
            if v > 0.55 + 0.07 * math.sin(u * 10):
                value = 95 + 30 * v  # Hills
            if 0.32 < u < 0.63 and 0.56 < v < 0.84:
                value = 200  # House
            if 0.26 < u < 0.69 and 0.40 + abs(u - 0.475) * 0.8 < v < 0.57:
                value = 65  # Roof
            if 0.43 < u < 0.51 and 0.67 < v < 0.84:
                value = 50  # Door
            if 0.55 < u < 0.60 and 0.63 < v < 0.71:
                value = 240  # Window
            if abs(u - 0.14) < 0.013 and 0.50 < v < 0.86:
                value = 60  # Tree trunk
            if (u - 0.14)**2 + (v - 0.47)**2 < 0.105**2:
                value = 80  # Tree crown
            pixels.append(min(255, max(0, int(value * darkness + 0.5))))
    return bytes(pixels)


def adjust(pixels, target=128):
    if not pixels or not 1 <= target <= 255:
        raise ValueError('Pixels must be nonempty and target must be in 1..255')
    mean = sum(pixels) / len(pixels)
    gain = target / mean if mean else 1.0
    output = bytes(min(255, int(value * gain + 0.5)) for value in pixels)
    return output, mean, gain


def write_png(path, pixels, width, height):
    if len(pixels) != width * height:
        raise ValueError('Pixel count does not match image dimensions')
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    scanlines = b''.join(b'\x00' + pixels[y*width:(y+1)*width] for y in range(height))
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 0, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b'')
    Path(path).write_bytes(png)
    return Path(path)


def reduction_svg(values):
    """Visualize the same half-stride additions used by the CUDA kernel."""
    if not values or len(values) & (len(values) - 1):
        raise ValueError('Reduction example needs a power-of-two count')
    rows = [list(values)]
    while len(rows[-1]) > 1:
        previous = rows[-1]
        half = len(previous) // 2
        rows.append([previous[i] + previous[i + half] for i in range(half)])
    width, height = 1040, 105 + len(rows) * 82
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="monospace" fill="#172033">',
             '<text x="24" y="28" font-size="19">Shared-memory reduction: 16 pixels become one block total</text>',
             '<text x="24" y="53" font-size="13">Each round adds the right half to the left half, then all threads synchronize.</text>']
    for step, row in enumerate(rows):
        y = 90 + step * 82
        label = 'Loaded' if step == 0 else f'+{len(row)} apart'
        parts.append(f'<text x="14" y="{y+25}" font-size="12">{label}</text>')
        for i, value in enumerate(row):
            x = 120 + i * 56
            parts.append(f'<rect x="{x}" y="{y}" width="50" height="42" rx="5" fill="#dbeafe" stroke="#2563eb"/>')
            parts.append(f'<text x="{x+7}" y="{y+26}" font-size="14">{value}</text>')
        if step < len(rows) - 1:
            parts.append(f'<text x="120" y="{y+63}" font-size="11">__syncthreads()</text>')
    parts.append('</g></svg>')
    return '\n'.join(parts)
