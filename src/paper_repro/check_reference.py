#!/usr/bin/env python3
"""Compare independently regenerated numeric CSV results against checked-in references.

Numbers are compared with tolerance for cross-platform NumPy differences.
Identifiers, categorical labels, row counts and column names must match exactly.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig",newline="") as fh:
        reader=csv.DictReader(fh)
        return list(reader.fieldnames or []),list(reader)


def equal_cell(a:str,b:str)->bool:
    try:
        x=float(a)
        y=float(b)
        if not (math.isfinite(x) and math.isfinite(y)):
            return a==b
        return math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-10)
    except (TypeError,ValueError):
        return a==b


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--actual",type=Path,required=True)
    args=ap.parse_args()
    files=sorted(x.name for x in args.reference.glob("*.csv"))
    if not files:
        raise AssertionError("No reference CSVs found")
    if sorted(x.name for x in args.actual.glob("*.csv"))!=files:
        raise AssertionError("Output CSV filenames differ")
    nrows=0
    for name in files:
        cols_expected, expected=read_rows(args.reference/name)
        cols_actual, actual=read_rows(args.actual/name)
        if cols_expected!=cols_actual:
            raise AssertionError(f"{name}: columns differ")
        if len(expected)!=len(actual):
            raise AssertionError(f"{name}: expected {len(expected)} rows, got {len(actual)}")
        for i,(a,b) in enumerate(zip(expected,actual),start=2):
            for c in cols_expected:
                if not equal_cell(a[c],b[c]):
                    raise AssertionError(f"{name} line {i}, field {c}: expected {a[c]!r}, got {b[c]!r}")
        nrows+=len(expected)
    print(f"REFERENCE COMPARISON PASSED: {len(files)} CSV files, {nrows} rows")


if __name__=="__main__":
    main()
