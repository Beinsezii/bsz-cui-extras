#! /usr/bin/env python3

from argparse import ArgumentParser, FileType
import re
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
parser.add_argument('-x', default=600)
parser.add_argument('-y', default=600)
parser.add_argument('files', nargs='+', type=FileType('r+'))

args = parser.parse_args()

pattern = re.compile(r'"pos": *\[\n? *(-?\d+),\n? *(-?\d+)')
#) -- my IDE gets confused so this helps

for f in args.files:
    data = json.load(f)
    minx, miny, maxx, maxy = 99999, 99999, -99999, -99999
    # 1st pass calc bounds
    for node in data["nodes"]:
        x, y = node["pos"]
        minx, miny, maxx, maxy = min(x, minx), min(y, miny), max(x, maxx), max(y, maxy)

    offset_x, offset_y = args.x - (maxx + minx) / 2, args.y - (maxy + miny) / 2

    for node in data["nodes"]:
        x, y = node["pos"]
        node["pos"][0] = int(roundint(x + offset_x, 10))
        node["pos"][1] = int(roundint(y + offset_y, 10))

    f.seek(0)
    json.dump(data, f, indent=2)
    f.truncate()
