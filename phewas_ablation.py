"""Reproduce the MESHAgents rows of the paper's Table 1 (Auto-PheWAS ablation study).

Runs our system in three configurations and reports the two Auto-PheWAS metrics for each:
    full            : the complete 3-stage protocol
    w/o Stage III   : single round, no iterative consensus  (ablation='no_stage3')
    w/o Mem. & Tool : no memory retrieval, no tool-evidence weighting in f_AP (ablation='no_mem_tools')

Metrics (paper §3):  Dependency Q(P) (lower = more independent + valid),  Coverage C(P) (higher = better).

NOTE: ablations differ only under the LLM engine (Stage III opinions). With MESH_DRY_RUN=1 the opinion
stub is identical across configs, so use a real model (or local TRE endpoint) for meaningful Table-1 rows.
Each config is a full protocol run (sequential LLM calls) -- expect several minutes per config.

Run from the repo root:  python phewas_ablation.py
"""
import os
import sys
import json
import asyncio
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "MESHAgents", "src"))

import llm_provider as P          # provider switch — MUST precede `import config` (ollama dummy-key shim)
import mesh_core
from config import OPENAI_API_KEY
from agents_bodycomp import (BodyCompChiefAgent, build_structures, detect_label,
                             FACTOR_CANDIDATES)

MERGED = os.path.join(HERE, "MESHAgents", "data", "merged_data.csv")
RESULTS = os.path.join(HERE, "MESHAgents", "results")
CLINICAL_FACTORS = ["age", "sex", "bmi", "weight", "waist_circumference"]
CONFIGS = [("full", None), ("w/o Stage III", "no_stage3"), ("w/o Mem. & Tool", "no_mem_tools")]


async def run_config(data, structures, name, ablation):
    chief = BodyCompChiefAgent(OPENAI_API_KEY, structures, CLINICAL_FACTORS)
    res = await chief.run_analysis(data, ablation=ablation)
    disc = res.get("discovered_phenotypes", [])
    metrics = res.get("auto_phewas_metrics", {})
    cons = res.get("consensus", {})
    return {"config": name, "ablation": ablation, "discovered": disc,
            "dependency_Q": metrics.get("dependency_Q"), "coverage_C": metrics.get("coverage_C"),
            "rounds_run": cons.get("rounds_run"), "converged": cons.get("converged")}


async def main():
    df = pd.read_csv(MERGED)
    structures = {k: [c for c in v if c in df.columns]
                  for k, v in build_structures(list(df.columns)).items()}
    structures = {k: v for k, v in structures.items() if v}
    if mesh_core.DRY_RUN:
        print("WARNING: MESH_DRY_RUN=1 -> ablations are degenerate (opinion stub identical). "
              "Use a real model for meaningful Table-1 rows.\n")

    rows = []
    for name, ablation in CONFIGS:
        print(f"running config: {name} ...")
        rows.append(await run_config(df, structures, name, ablation))

    print(f"\n{'config':18s} {'Dependency Q':>14s} {'Coverage C':>12s} {'rounds':>8s}")
    print("-" * 56)
    for r in rows:
        q = f"{r['dependency_Q']:.3f}" if r["dependency_Q"] is not None else "n/a"
        c = f"{r['coverage_C']:.3f}" if r["coverage_C"] is not None else "n/a"
        print(f"{r['config']:18s} {q:>14s} {c:>12s} {str(r['rounds_run']):>8s}")

    out = os.path.join(RESULTS, "phewas_ablation.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")
    print("(Paper Table 1, MESHAgents rows: full has best/lowest Dependency and strong Coverage; "
          "w/o Stage III and w/o Mem.&Tool degrade.)")


if __name__ == "__main__":
    asyncio.run(main())
