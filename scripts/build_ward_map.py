"""Build web/assets/data/ward_map.json: real ward shapes for the dashboard map (offline, no tiles).

    python -m scripts.build_ward_map ADM2.geojson ADM3.geojson ADM4.geojson

Inputs are the *simplified* geoBoundaries files for Bangladesh (gbOpen BGD ADM2/ADM3/ADM4,
source: Bangladesh Bureau of Statistics + OCHA ROAP, licence CC BY 3.0 IGO), downloaded by hand from
https://www.geoboundaries.org. This script never touches the network: it only reads local files.

What the public data has, and the assumptions we make (also shown on the map page):
- Chattogram: shapes for all 41 city wards ("Ward No-01".."Ward No-41"). Dataset CCC_Ward_N -> ward N.
- Dhaka: shapes use the pre-2011 Dhaka City Corporation numbering (wards 1-92). After the split, Dhaka
  North kept the old numbers 1-20, and Dhaka South numbers its wards in old order starting at old 21,
  skipping the old wards that went north (37-47). So DNCC_Ward_N -> old N, and DSCC_Ward_N -> old 20+N
  for N <= 16, old 31+N for N >= 17. The dataset gives only IDs, so this mapping is an assumption.
- Gazipur, Narayanganj: these city corporations were formed after the census frame, so no ward shapes
  exist publicly. We export the city's core area outline only; the page lists their wards beside it.
Wards split into "(part)" pieces are merged. Points are simplified (Douglas-Peucker) and rounded.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "assets" / "data" / "ward_map.json"
TOL = 0.00015          # degrees (~15 m): simplification tolerance
DIGITS = 5

DISTRICT = {"Dhaka": "Dhaka", "Chittagong": "Chattogram", "Gazipur": "Gazipur", "Narayanganj": "Narayanganj"}
CITY_AREAS = {  # ADM4 units that make up each city's core (no ward shapes exist for these two)
    "Gazipur": {"Gazipur Paurashava", "Tongi Paurashava"},
    "Narayanganj": {"Narayanganj Paurashava", "Kadam Rasul Paurashava"},
}


def dncc_old(n: int) -> int:
    return n


def dscc_old(n: int) -> int:
    return 20 + n if n <= 16 else 31 + n


def polys(geom: dict) -> list:
    return geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]


def centroid(geom: dict) -> tuple[float, float]:
    ring = max(polys(geom), key=lambda p: len(p[0]))[0]
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        c = x1 * y2 - x2 * y1
        a += c; cx += (x1 + x2) * c; cy += (y1 + y2) * c
    return (cx / (3 * a), cy / (3 * a)) if a else tuple(ring[0])


def inside(pt, geom) -> bool:
    x, y = pt
    for p in polys(geom):
        ring, hit = p[0], False
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                hit = not hit
        if hit:
            return True
    return False


def simplify(pts: list, tol: float) -> list:
    """Douglas-Peucker. A closed ring is split at its farthest point first (start == end otherwise)."""
    if len(pts) < 4:
        return pts
    if pts[0] == pts[-1]:
        x0, y0 = pts[0]
        k = max(range(len(pts)), key=lambda i: (pts[i][0] - x0) ** 2 + (pts[i][1] - y0) ** 2)
        return _dp(pts[:k + 1], tol) + _dp(pts[k:], tol)[1:]
    return _dp(pts, tol)


def _dp(pts: list, tol: float) -> list:
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = pts[i], pts[j]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
        best, idx = 0.0, -1
        for k in range(i + 1, j):
            d = abs(dy * pts[k][0] - dx * pts[k][1] + x2 * y1 - y2 * x1) / norm
            if d > best:
                best, idx = d, k
        if best > tol:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(pts, keep) if k]


def clean(geom: dict, tol: float = TOL) -> list:
    """Outer rings only, simplified and rounded: [[[lon, lat], ...], ...]."""
    out = []
    for p in polys(geom):
        ring = simplify(p[0], tol)
        if len(ring) >= 4:
            out.append([[round(x, DIGITS), round(y, DIGITS)] for x, y in ring])
    return out


def main(adm2: str, adm3: str, adm4: str) -> None:
    load = lambda p: json.loads(Path(p).read_text())["features"]
    a2, a3, a4 = load(adm2), load(adm3), load(adm4)
    districts = {DISTRICT[f["properties"]["shapeName"]]: f for f in a2 if f["properties"]["shapeName"] in DISTRICT}

    # every "Ward No-NN" shape inside one of our districts, grouped by (district, number); parts merged
    wards: dict[tuple[str, int], list] = {}
    for f in a4:
        m = re.match(r"Ward No-(\d+)", f["properties"]["shapeName"])
        if not m:
            continue
        c = centroid(f["geometry"])
        d = next((name for name, g in districts.items() if inside(c, g["geometry"])), None)
        if d in ("Dhaka", "Chattogram"):
            wards.setdefault((d, int(m.group(1))), []).extend(clean(f["geometry"]))

    mapped, used = {}, set()
    for n in range(1, 21):
        mapped[f"DNCC_Ward_{n:02d}"] = ("Dhaka", dncc_old(n))
        mapped[f"DSCC_Ward_{n:02d}"] = ("Dhaka", dscc_old(n))
    for n in range(1, 11):
        mapped[f"CCC_Ward_{n:02d}"] = ("Chattogram", n)
    shapes = {}
    for ward_id, key in mapped.items():
        if key not in wards:
            raise SystemExit(f"No shape for {ward_id} ({key})")
        shapes[ward_id] = {"source_ward": key[1], "rings": wards[key]}
        used.add(key)
    context = {city: [r for (d, _), rings in sorted(wards.items()) if d == city and (d, _) not in used for r in rings]
               for city in ("Dhaka", "Chattogram")}

    areas = {}
    for city, names in CITY_AREAS.items():
        rings = [r for f in a4 if f["properties"]["shapeName"] in names for r in clean(f["geometry"])]   # names are unique
        if not rings:
            raise SystemExit(f"No area shape for {city}")
        areas[city] = rings

    out = {
        "source": "Bangladesh Bureau of Statistics and OCHA ROAP, via geoBoundaries (gbOpen BGD ADM2-ADM4), "
                  "CC BY 3.0 IGO. Simplified.",
        "assumptions": {
            "Dhaka North": "DNCC_Ward_N is drawn on old Dhaka City Corporation ward N (N = 1-20).",
            "Dhaka South": "DSCC_Ward_N is drawn on old ward 20+N (N <= 16) or 31+N (N >= 17), "
                           "the old wards numbered in order after Dhaka North's share.",
            "Chattogram": "CCC_Ward_N is drawn on Chattogram ward N.",
            "Gazipur": "No public ward shapes; city core area only (Gazipur and Tongi paurashavas).",
            "Narayanganj": "No public ward shapes; city core area only (Narayanganj and Kadam Rasul paurashavas).",
        },
        "wards": shapes,
        "context": context,                  # the city's other wards, drawn grey for orientation
        "areas": areas,
        "districts": {k: clean(v["geometry"], 0.002) for k, v in districts.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB, {len(shapes)} wards)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
