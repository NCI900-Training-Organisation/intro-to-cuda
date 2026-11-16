"""Draw labelled CUDA thread grids as SVG images; uses only Python's standard library.

Without --csv, draw the mathematical index mapping (no GPU execution).
With --csv, validate and draw records produced by thread_indexing.cu.
"""
import argparse
import csv
from html import escape
from pathlib import Path


def reference_records(gx, gy, bx, by):
    for y in range(gy * by):
        for x in range(gx * bx):
            yield dict(global_=y * gx * bx + x, local=(y % by) * bx + x % bx,
                       bx=x // bx, by=y // by, tx=x % bx, ty=y % by, x=x, y=y)


def draw(gx, gy, bx, by, output, csv_path=None, records=None):
    expected = list(reference_records(gx, gy, bx, by))
    if csv_path and records is not None:
        raise ValueError("Provide CSV input or notebook records, not both")
    if csv_path:
        with Path(csv_path).open(newline='') as source:
            records = [{('global_' if k == 'global' else k): int(v)
                        for k, v in row.items()} for row in csv.DictReader(source)]
        records.sort(key=lambda row: row['global_'])
        if records != expected:
            raise ValueError('GPU records do not match the expected thread coordinates and IDs')
        origin = 'GPU-recorded indices (validated against the indexing formula)'
    elif records is not None:
        records = sorted(records, key=lambda row: row['global_'])
        if records != expected:
            raise ValueError('Notebook records do not match the selected grid/block dimensions')
        origin = 'Notebook variable: indices calculated from formulas; no GPU execution'
    else:
        records = expected
        origin = 'Reference diagram calculated from indexing formulas; no GPU execution'
    cw, ch, margin, top = 108, 76, 32, 120
    width, height = gx * bx * cw + margin * 2, gy * by * ch + top + 48
    width = max(width, 740)
    palette = ['#dbeafe', '#ffedd5', '#dcfce7', '#f3e8ff', '#fce7f3', '#cffafe']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="monospace" fill="#172033">']
    def text(x, y, value, size=13):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(value))}</text>')
    text(margin, 28, f'Grid ({gx}, {gy}) blocks | Block ({bx}, {by}) threads', 17)
    text(margin, 52, 'G = global ID; L = local ID within block; B = block (x,y)')
    text(margin, 74, 'T = threadIdx (x,y). Thick outlines mark blocks. x increases right; y down.')
    text(margin, 96, origin, 11)
    for r in records:
        x, y = margin + r['x'] * cw, top + r['y'] * ch
        color = palette[(r['by'] * gx + r['bx']) % len(palette)]
        parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{color}" stroke="#94a3b8"/>')
        text(x + 7, y + 20, f"G {r['global_']} | L {r['local']}")
        text(x + 7, y + 42, f"B ({r['bx']},{r['by']})")
        text(x + 7, y + 62, f"T ({r['tx']},{r['ty']})")
    for block_y in range(gy):
        for block_x in range(gx):
            parts.append(f'<rect x="{margin + block_x * bx * cw}" y="{top + block_y * by * ch}" width="{bx*cw}" height="{by*ch}" fill="none" stroke="#172033" stroke-width="3"/>')
    text(margin, height - 18, f'{gx * gy} blocks x {bx * by} threads = {len(records)} threads')
    parts.append('</g></svg>')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(parts) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dimensions', type=int, nargs=4, metavar='DIM', help='grid_x grid_y block_x block_y')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    if any(d < 1 or d > 16 for d in args.dimensions):
        parser.error('Each dimension must be from 1 to 16')
    print(draw(*args.dimensions, args.output, args.csv))


if __name__ == '__main__':
    main()
