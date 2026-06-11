"""Reproduce the 'Independent LLMs (zero-shot CoT)' rows of the paper's Table 1.

A single LLM is given the PheWAS task definition and the candidate phenotype/factor lists, and asked
in one zero-shot Chain-of-Thought pass to select the disease-associated phenotypes + confounders.
We then score its output with the two Auto-PheWAS metrics (Dependency Q, Coverage C) and place it next
to the MESHAgents consensus set -- directly reproducing the paper's comparison (single LLMs show higher
dependency / lower coverage from hallucination + catastrophic forgetting; MESHAgents is best).

Models (paper: GPT-3.5, GPT-4o-mini, Claude-3.5). Configure via MESH_BASELINE_MODELS (comma-separated
OpenAI ids). Claude is included automatically iff ANTHROPIC_API_KEY is set and the `anthropic` SDK is
installed; otherwise it is skipped with a note (it is a different vendor's model).

Run from the MESHAgents/ directory (so .env loads):  python ../phewas_baselines.py
"""
import os
import re
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "MESHAgents", "src"))

import llm_provider as P          # provider switch — MUST precede `import config` (ollama dummy-key shim)
import mesh_core
from config import OPENAI_API_KEY
from agents_bodycomp import build_structures, FACTOR_CANDIDATES, DISCOVER_K, detect_label

MERGED = os.path.join(HERE, "MESHAgents", "data", "merged_data.csv")
RESULTS = os.path.join(HERE, "MESHAgents", "results")
# Baseline models: OpenAI list by default; in ollama mode default to the single local chat model.
_DEFAULT_BASELINE = P.chat_model() if P.is_ollama() else "gpt-4o-mini,gpt-3.5-turbo,gpt-5-mini"
OPENAI_MODELS = os.getenv("MESH_BASELINE_MODELS", _DEFAULT_BASELINE).split(",")

TASK = (
    "You are performing a phenome-wide association study (PheWAS). From the candidate body-composition "
    "phenotypes below, select the {k} phenotypes most strongly and meaningfully associated with the "
    "disease '{disease}', and list the non-imaging factors that are confounders (associated with both the "
    "phenotypes and the disease). Think step by step (chain-of-thought), then give your final answer.\n\n"
    "Candidate phenotypes ({n}):\n{phenos}\n\nCandidate factors:\n{factors}\n\n"
    "Return ONLY JSON as the final line: "
    "{{\"phenotypes\": [<exactly {k} phenotype names from the candidates>], \"confounders\": [<factors>]}}"
)


def _parse(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def call_openai(model, system, user):
    client = P.chat_client()                              # provider-aware (OpenAI or local Ollama)
    try:
        r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                           messages=[{"role": "system", "content": system},
                                                     {"role": "user", "content": user}])
    except Exception:
        r = client.chat.completions.create(model=model,
                                           messages=[{"role": "system", "content": system},
                                                     {"role": "user", "content": user}])
    return _parse(r.choices[0].message.content or "")


def call_anthropic(model, system, user):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(model=model, max_tokens=1024, system=system,
                               messages=[{"role": "user", "content": user}])
    return _parse("".join(b.text for b in r.content if getattr(b, "type", "") == "text"))


def score(name, selected, phenos, data):
    """Compute Auto-PheWAS Q/C on a model's selected phenotype set."""
    valid = [p for p in selected if p in phenos]
    return {"model": name, "n_selected": len(selected), "n_valid": len(valid),
            "hallucinated": [p for p in selected if p not in phenos],
            "dependency_Q": mesh_core.dependency(selected, data, phenotype_universe=phenos),
            "coverage_C": mesh_core.coverage(selected, phenos), "phenotypes": selected}


def main():
    df = pd.read_csv(MERGED)
    phenos = [c for c in df.columns if c.startswith(("body_comp_", "total_scan_"))]
    factors = [f for f in FACTOR_CANDIDATES if f in df.columns]
    disease = detect_label(df.columns) or "diabetes"
    system = "You are an expert in cardiometabolic imaging phenotypes and statistical association studies."
    user = TASK.format(k=DISCOVER_K, disease=disease, n=len(phenos),
                       phenos="\n".join(phenos), factors=", ".join(factors))

    rows = []
    for model in [m.strip() for m in OPENAI_MODELS if m.strip()]:
        print(f"querying {model} (zero-shot CoT)...")
        try:
            out = call_openai(model, system, user)
            sel = [p for p in out.get("phenotypes", [])][:DISCOVER_K]
            rows.append(score(model, sel, phenos, df))
        except Exception as e:
            print(f"  {model} failed: {e}")

    # Claude-3.5 only if a key + SDK are present (different vendor)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            cmodel = os.getenv("MESH_CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
            print(f"querying {cmodel} (zero-shot CoT)...")
            out = call_anthropic(cmodel, system, user)
            rows.append(score(cmodel, [p for p in out.get("phenotypes", [])][:DISCOVER_K], phenos, df))
        except Exception as e:
            print(f"  Claude skipped: {e}")
    else:
        print("(Claude-3.5 skipped: set ANTHROPIC_API_KEY + `pip install anthropic` to include it.)")

    # MESHAgents consensus set for side-by-side comparison
    disc = sorted(glob.glob(os.path.join(RESULTS, "discovered_phenotypes_*.json")))
    if disc:
        mesh = json.load(open(disc[-1])).get("discovered_phenotypes", [])
        rows.append(score("MESHAgents (ours)", mesh, phenos, df))

    print(f"\n{'model':28s} {'Dependency Q':>13s} {'Coverage C':>12s} {'valid/sel':>10s}")
    print("-" * 66)
    for r in rows:
        q = f"{r['dependency_Q']:.3f}" if r["dependency_Q"] is not None else "n/a"
        c = f"{r['coverage_C']:.3f}" if r["coverage_C"] is not None else "n/a"
        print(f"{r['model']:28s} {q:>13s} {c:>12s} {r['n_valid']}/{r['n_selected']:>3d}")

    out = os.path.join(RESULTS, "phewas_baselines.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
