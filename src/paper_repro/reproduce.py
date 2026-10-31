#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce numerical findings using PUBLIC CSV files only; no images required.

Use: python -m src.paper_repro.reproduce --data data/paper_repro/npj_v1
Outputs are generated under the dataset's reference_results/ folder by default.
Requires numpy. Never requires raw pages, cropped glyphs, OCR or review-state JSON.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["aspect_ratio", "face_ratio", "black_white_ratio"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def checked_float(x: Any) -> float:
    z = float(x)
    if not math.isfinite(z):
        raise ValueError(f"Nonfinite metric {x!r}")
    return z


def stable_seed(tag: str, x: list[float], y: list[float]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(tag.encode("utf-8"))
    h.update(b"\0")
    h.update(np.asarray(x, dtype=float).tobytes())
    h.update(b"\0")
    h.update(np.asarray(y, dtype=float).tobytes())
    return int.from_bytes(h.digest(), "little", signed=False) & 0xFFFFFFFF


def ranks(values: list[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    out = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[order[j]] == v[order[i]]:
            j += 1
        out[order[i:j]] = (i+j+1)/2.0
        i = j
    return out


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_spearman(x: list[float], y: list[float], rho: float) -> tuple[float, str]:
    n = len(x)
    if n <= 9:
        hits = total = 0
        for idx in itertools.permutations(range(n)):
            rp = spearman(x, [y[i] for i in idx])
            hits += math.isfinite(rp) and abs(rp) >= abs(rho)-1e-12
            total += 1
        return hits / total, "exact_permutation"
    n_perm = 5000
    rng = np.random.default_rng(stable_seed("rho_perm", x, y))
    ys = np.asarray(y, dtype=float)
    hits = 1
    for _ in range(n_perm):
        rp = spearman(x, ys[rng.permutation(n)].tolist())
        if math.isfinite(rp) and abs(rp) >= abs(rho)-1e-12:
            hits += 1
    return hits / (n_perm+1), f"monte_carlo_{n_perm}"


def trend(x: list[float], y: list[float]) -> dict:
    rho = spearman(x, y)
    if not math.isfinite(rho):
        raise ValueError("Invalid Spearman result")
    p, method = permutation_spearman(x, y, rho)
    # Theil-Sen: median of all pairwise slopes with distinct dates (not causal).
    slopes = [(y[j]-y[i])/(x[j]-x[i]) for i in range(len(x)) for j in range(i+1,len(x)) if x[j] != x[i]]
    slope = float(np.median(slopes)) if slopes else float("nan")
    rng = np.random.default_rng(stable_seed("rho", x, y))
    bootstrap = []
    for _ in range(400):
        ix = rng.integers(0, len(x), len(x))
        v = spearman(np.asarray(x)[ix].tolist(), np.asarray(y)[ix].tolist())
        if math.isfinite(v):
            bootstrap.append(v)
    low, high = (np.quantile(bootstrap, [.025,.975]) if bootstrap else [float("nan"),float("nan")])
    return {
        "n_books": len(x), "rho": rho, "p_raw": p, "p_method": method,
        "theil_sen_per_century": slope*100, "rho_ci_lower":float(low),
        "rho_ci_upper":float(high),
    }


def bh_adjust(values: list[float]) -> list[float]:
    n = len(values)
    ordered = sorted(range(n), key=lambda i: values[i])
    output = [1.0]*n
    level = 1.0
    for j in range(n-1, -1, -1):
        i = ordered[j]
        level = min(level, min(1.0, values[i]*n/(j+1)))
        output[i] = level
    return output


def two_group_exact_difference(a: list[float], b: list[float]) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = float(np.mean(b)-np.mean(a))
    x = np.concatenate([a,b])
    hits = total = 0
    for ai in itertools.combinations(range(len(x)),len(a)):
        mask = np.zeros(len(x),dtype=bool)
        mask[list(ai)] = True
        delta = float(np.mean(x[~mask])-np.mean(x[mask]))
        hits += abs(delta) >= abs(obs)-1e-12
        total += 1
    return obs, hits/total


def temporal_regional_interaction_exact(values: dict[str,list[float]]) -> tuple[float,float,float,float]:
    # Null: years/period are exchangeable WITHIN region. Exact 8008 x 20 arrangements.
    lm = values["liangzhe_middle"]; ll = values["liangzhe_late"]
    fm = values["fujian_middle"]; fl = values["fujian_late"]
    gap_mid = float(np.mean(fm)-np.mean(lm))
    gap_late = float(np.mean(fl)-np.mean(ll))
    observed = gap_late-gap_mid
    def null_period_deltas(mid: list[float],late: list[float]) -> np.ndarray:
        x = np.asarray(mid+late,dtype=float);n=len(mid)
        out=[]
        for choice in itertools.combinations(range(len(x)),n):
            mask = np.zeros(len(x),dtype=bool)
            mask[list(choice)]=True
            out.append(float(np.mean(x[~mask])-np.mean(x[mask])))
        return np.asarray(out)
    # interaction = (F_late - F_mid) - (L_late - L_mid)
    l_null=null_period_deltas(lm,ll)
    f_null=null_period_deltas(fm,fl)
    interaction_null=f_null[:,None]-l_null[None,:]
    p=float(np.mean(np.abs(interaction_null)>=abs(observed)-1e-12))
    return gap_mid,gap_late,observed,p


def load_data(root: Path) -> tuple[dict,dict,dict,dict]:
    plan=json.loads((root/"analysis_plan.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for file,expected in manifest["file_sha256"].items():
        if sha(root/file)!=expected:
            raise AssertionError(f"Publication dataset hash mismatch: {file}")
    books=read_csv(root/"books.csv")
    instances=read_csv(root/"instance_metrics.csv")
    character_rows=read_csv(root/"book_character_metrics.csv")
    if (len(books),len(instances),len(character_rows)) != (38,22468,9043):
        raise AssertionError("Frozen dataset row-count mismatch")
    metadata={r["book_id"]:r for r in books}
    if len(metadata)!=38:
        raise AssertionError("Duplicate book ids")
    groups:dict[str,dict[str,list[dict]]]=collections.defaultdict(lambda:collections.defaultdict(list))
    seen=set()
    for entry in instances:
        key=(entry["book_id"],entry["char"],entry["instance_id"])
        if key in seen:
            raise AssertionError(f"Duplicate instance: {key}")
        seen.add(key)
        if entry["book_id"] not in metadata:
            raise AssertionError("Instance book id absent from metadata")
        groups[entry["book_id"]][entry["char"]].append(entry)
    ntypes=len({c for m in groups.values() for c in m})
    if ntypes!=317 or sum(len(z) for z in groups.values())!=9043:
        raise AssertionError("Frozen numeric char coverage mismatch")
    agg=aggregate(groups,cap3=False)
    # Verify the public instance-level table can independently recreate all book-char metrics.
    for r in character_rows:
        b,c=r["book_id"],r["char"]
        if int(r["n_instances"])!=len(groups[b][c]):
            raise AssertionError(f"Character count disagreement {b}/{c}")
        for m in METRICS:
            if not np.isclose(agg[b][c][m],checked_float(r[m]),rtol=0,atol=1e-12):
                raise AssertionError(f"Aggregation mismatch {b}/{c}/{m}")
    for book,meta in metadata.items():
        if sum(len(entries) for entries in groups[book].values()) != int(meta["n_instances"]):
            raise AssertionError(f"Book instance count mismatch: {book}")
        if len(groups[book]) != int(meta["n_characters"]):
            raise AssertionError(f"Book character coverage mismatch: {book}")
    if len(plan["temporal_groups"]["liangzhe_near_ou_18"])!=18 or len(plan["temporal_groups"]["fujian_6"])!=6:
        raise AssertionError("Unexpected series sizes")
    return plan,metadata,groups,agg


def instance_sort_key(v: dict) -> tuple:
    def num(v: str) -> int:
        try:return int(v)
        except:return 10**8
    return (num(v.get("volume","")),v.get("page",""),num(v.get("char_index","")),v["instance_id"])


def aggregate(groups: dict, cap3: bool) -> dict:
    res={}
    for book,charmap in groups.items():
        res[book]={}
        for char,entries in charmap.items():
            subset = sorted(entries,key=instance_sort_key)[:3] if cap3 else entries
            res[book][char]={m:float(np.mean([checked_float(e[m]) for e in subset])) for m in METRICS}
    return res


def book_group_stats(books: list[str],metric: str, agg: dict,q:float,metadata:dict) -> tuple[int,list[dict]]:
    if not books:
        raise ValueError("Empty comparison group")
    common=set.intersection(*(set(agg[b]) for b in books))
    if not common:
        raise ValueError("No shared characters")
    out=[]
    for b in books:
        arr=np.asarray([agg[b][c][metric] for c in sorted(common)],dtype=float)
        if q>0 and len(arr)>=4:
            lo,hi=np.quantile(arr,[q,1.0-q]);arr=arr[(arr>=lo)&(arr<=hi)]
        out.append({"book_id":b,"year":int(metadata[b]["year_nominal"]),
                    "region":metadata[b]["region"],"metric":metric,
                    "common_characters":len(common),"retained_characters":len(arr),
                    "book_mean":float(np.mean(arr)),
                    "book_dispersion":float(np.std(arr,ddof=0))})
    return len(common),out


def compute_temporal(plan:dict,meta:dict,agg:dict,cap3:bool=False) -> tuple[list[dict],list[dict]]:
    rows=[];tests=[]; q=float(plan["trim_quantile"])
    for group,books in plan["temporal_groups"].items():
        books=sorted(books,key=lambda b:(int(meta[b]["year_nominal"]),int(meta[b]["number"])))
        for metric in METRICS:
            _,r=book_group_stats(books,metric,agg,q,meta)
            for b in r:
                rows.append({"comparison":group,"cap3":cap3,**b})
            for moment,col in [("mean","book_mean"),("dispersion","book_dispersion")]:
                x=[float(v["year"]) for v in r]
                y=[float(v[col]) for v in r]
                res=trend(x,y)
                tests.append({"comparison":group,"cap3":cap3,"metric":metric,"statistic":moment,
                              "shared_characters":r[0]["common_characters"],
                              "retained_min":min(v["retained_characters"] for v in r),
                              "retained_max":max(v["retained_characters"] for v in r),**res})
    # BH over 12 temporal tests; not a claim that all tests are independent.
    adj=bh_adjust([t["p_raw"] for t in tests])
    for t,p in zip(tests,adj):t["p_BH_12_tests"]=p
    return rows,tests


def compute_regional(plan:dict,meta:dict,agg:dict,cap3:bool=False) -> tuple[list[dict],list[dict],list[dict],list[dict]]:
    q=float(plan["trim_quantile"])
    rows=[];summary=[];contrasts=[];fixed=[]
    for period in plan["region_comparisons"]:
        books=[b for b in meta if meta[b]["period_nominal"]==period]
        present=sorted({meta[b]["region"] for b in books})
        for metric in METRICS:
            n,rs=book_group_stats(books,metric,agg,q,meta)
            for x in rs: rows.append({"comparison":period,"cap3":cap3,**x})
            by=collections.defaultdict(list)
            for x in rs:by[x["region"]].append(x)
            for reg in present:
                vals=by[reg]
                summary.append({"period":period,"cap3":cap3,"metric":metric,"region":reg,
                                "n_books":len(vals),"shared_characters":n,
                                "mean_book_mean":float(np.mean([v["book_mean"] for v in vals])),
                                "mean_book_dispersion":float(np.mean([v["book_dispersion"] for v in vals]))})
            for a,b in itertools.combinations(present,2):
                aa=[x["book_mean"] for x in by[a]]
                bb=[x["book_mean"] for x in by[b]]
                diff,p=two_group_exact_difference(aa,bb)
                contrasts.append({"period":period,"cap3":cap3,"metric":metric,
                                  "region_a":a,"region_b":b,"shared_characters":n,
                                  "n_a":len(aa),"n_b":len(bb),"difference_b_minus_a":diff,
                                  "p_exact_two_sided":p})
    # A different, FIXED set of shared characters is necessary to compare regional
    # contrasts ACROSS historical periods; no separate 89 vs 99 pools allowed here.
    c=plan["fixed_common_pool_region_comparison"]
    books=[b for b in meta if meta[b]["region"] in c["regions"]
           and meta[b]["period_nominal"] in c["periods"]]
    for metric in METRICS:
        n,rs=book_group_stats(books,metric,agg,q,meta)
        four={k:[] for k in ("liangzhe_middle","liangzhe_late","fujian_middle","fujian_late")}
        for x in rs:
            region="liangzhe" if x["region"]=="两浙地区" else "fujian"
            period=meta[x["book_id"]]["period_nominal"]
            four[region+"_"+period].append(x["book_mean"])
        gap_mid,gap_late,delta,p=temporal_regional_interaction_exact(four)
        fixed.append({"cap3":cap3,"metric":metric,"shared_characters":n,
                      "n_liangzhe_middle":len(four["liangzhe_middle"]),
                      "n_liangzhe_late":len(four["liangzhe_late"]),
                      "n_fujian_middle":len(four["fujian_middle"]),
                      "n_fujian_late":len(four["fujian_late"]),
                      "mid_fujian_minus_liangzhe":gap_mid,"late_fujian_minus_liangzhe":gap_late,
                      "change_in_gap":delta,"p_exact_exchangeable_within_region":p})
    return rows,summary,contrasts,fixed


def year_sensitivity(plan:dict,meta:dict,agg:dict) -> list[dict]:
    fixed_books=plan["temporal_groups"]["liangzhe_near_ou_18"]
    key="26_1195_中庸辑略"
    outputs=[]
    # Hold the common 109-character pool fixed across all the dating scenarios.
    for metric in METRICS:
        _,group=book_group_stats(fixed_books,metric,agg,float(plan["trim_quantile"]),meta)
        by={r["book_id"]:r for r in group}
        for scenario in plan["year_uncertainty"]["中庸辑略_scenarios"]:
            books=[b for b in fixed_books if not (scenario=="exclude" and b==key)]
            x=[float(int(scenario)) if b==key else float(meta[b]["year_nominal"]) for b in books]
            for moment,col in [("mean","book_mean"),("dispersion","book_dispersion")]:
                vals=[by[b][col] for b in books]
                t=trend(x,vals)
                outputs.append({"metric":metric,"statistic":moment,"dating_scenario":scenario,
                                "common_pool_fixed_at":group[0]["common_characters"],**t})
    return outputs


def emit(root:Path,output:Path) -> dict:
    plan,meta,groups,agg=load_data(root)
    cap=aggregate(groups,cap3=True)
    tbook,tstats=compute_temporal(plan,meta,agg)
    rbook,rsum,rcontrast,rfixed=compute_regional(plan,meta,agg)
    tbook_cap,tstats_cap=compute_temporal(plan,meta,cap,cap3=True)
    rbook_cap,rsum_cap,rcontrast_cap,rfixed_cap=compute_regional(plan,meta,cap,cap3=True)
    dating=year_sensitivity(plan,meta,agg)
    report={
        "temporal_book_values.csv":tbook,
        "temporal_trends.csv":tstats,
        "regional_book_values.csv":rbook,
        "regional_summaries.csv":rsum,
        "regional_pairwise_tests.csv":rcontrast,
        "regional_fixed_77_contrasts.csv":rfixed,
        "dating_sensitivity.csv":dating,
        "cap3_temporal_trends.csv":tstats_cap,
        "cap3_regional_fixed_contrasts.csv":rfixed_cap,
    }
    for name,rows in report.items():
        if not rows:raise AssertionError(f"No rows for {name}")
        write_csv(output/name,rows,list(rows[0]))
    # Quality gates on fundamental data claims. Tests are not used to maximize significance.
    assert len(meta)==38 and len(groups)==38
    assert len(tstats)==12 and len(rfixed)==3
    assert {x["shared_characters"] for x in rfixed}=={77}
    assert {x["shared_characters"] for x in tstats if x["comparison"]=="liangzhe_near_ou_18"}=={109}
    assert {x["shared_characters"] for x in tstats if x["comparison"]=="fujian_6"}=={105}
    return {
        "books":38,"instances":22468,"book_characters":9043,
        "near_ou_books":18,"fujian_books":6,"near_ou_common":109,"fujian_common":105,
        "fixed_regional_common":77,
        "temporal_tests":12,
        "outputs":{k:sha(output/k) for k in report},
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=ROOT/"data/paper_repro/npj_v1")
    ap.add_argument("--out",type=Path,default=None)
    args=ap.parse_args()
    root=args.data.resolve()
    out=(args.out or (root/"reference_results")).resolve()
    out.mkdir(parents=True,exist_ok=True)
    summary=emit(root,out)
    (out/"verification.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("NUMERICAL REPRODUCTION PASSED:",json.dumps({k:v for k,v in summary.items() if k!="outputs"},ensure_ascii=False),flush=True)
    with (out/"temporal_trends.csv").open(encoding="utf-8") as f:
        print("TIME TRENDS:\n"+f.read()[:4000])
    with (out/"regional_fixed_77_contrasts.csv").open(encoding="utf-8") as f:
        print("FIXED-CHAR REGION CONTRASTS:\n"+f.read())


if __name__=="__main__":
    main()
