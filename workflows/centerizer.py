#! /usr/bin/env python3

from argparse import ArgumentParser, FileType
import json

def roundint(n: int, step: int) -> int:
    inv=False
    if n < 0:
        inv=True
        n = abs(n)
    if n % step >= step/2:
        result = n + step - (n % step)
    else:
        result = n - (n % step)
    if inv:
        result = -result
    return result

parser = ArgumentParser()
parser.add_argument('-x', default=600, help="Origin point X. Default 600")
parser.add_argument('-y', default=600, help="Origin point Y. Default 600")
parser.add_argument('-s', '--snap', default=10, help="Snap size")
parser.add_argument('files', nargs='+', type=FileType('r+'), help="List of .json files to centerize")

args = parser.parse_args()

for f in args.files:
    data = json.load(f)
    snap = int(args.snap)
    minx, miny, maxx, maxy = 99999, 99999, -99999, -99999
    # 1st pass calc bounds
    for node in data["nodes"]:
        x, y = node["pos"]
        minx, miny, maxx, maxy = min(x, minx), min(y, miny), max(x, maxx), max(y, maxy)

    for group in data["groups"]:
        x, y = group["bounding"][:2]
        minx, miny, maxx, maxy = min(x, minx), min(y, miny), max(x, maxx), max(y, maxy)

    offset_x, offset_y = args.x - (maxx + minx) / 2, args.y - (maxy + miny) / 2

    for node in data["nodes"]:
        x, y = node["pos"]
        node["pos"][0] = int(roundint(x + offset_x, snap))
        node["pos"][1] = int(roundint(y + offset_y, snap))

    for group in data["groups"]:
        x, y = group["bounding"][:2]
        group["bounding"][0] = int(roundint(x + offset_x, snap))
        group["bounding"][1] = int(roundint(y + offset_y, snap))

    f.seek(0)
    json.dump(data, f, indent=2)
    f.truncate()
