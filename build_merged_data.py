"""Adapter: build MESHAgents-format merged_data.csv from HPP/synthetic tables via pheno_io.

Output: MESHAgents/data/merged_data.csv  (body-composition phenotypes X + confounders C + binary label)
        MESHAgents/data/structures.json  (region -> specialist phenotype groups)

Local: reads synthetic parquet (pheno_io.USE_SYNTHETIC=True).
HPP VM: flip pheno_io.USE_SYNTHETIC=False -> same code uses PhenoLoader.
"""
import os
import re
import json
import argparse
import pandas as pd
from pheno_io import get_df

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "MESHAgents", "data")
STAGE = "00_00_visit"

# case = 1, control = 0 per disease (confirm against the real curated_phenotype categories on the VM)
POSITIVE = {
    "diabetes":            lambda v: v in {"diabetes", "diabetes_medication"},
    "hypertension":        lambda v: "hypertens" in v.lower() and "non" not in v.lower(),
    "abdominal_adiposity": lambda v: "risk" in v.lower(),
    "mafld":               lambda v: "normal" not in v.lower(),
    "osteoporosis":        lambda v: "osteoporosis" in v.lower() or "osteopenia" in v.lower(),
}


def baseline(df):
    d = df.reset_index()
    d = d[d["research_stage"] == STAGE].drop_duplicates("participant_id")
    return d.set_index("participant_id")


# columns that look like they contain a token but must never be picked as a confounder
_RESERVED = {"research_stage", "array_index", "participant_id", "cohort", "collection_date",
             "collection_timestamp", "timezone"}


def pick_col(columns, token):
    """Select the column for a confounder `token` (e.g. 'age', 'sex') without matching
    substrings inside reserved keys like 'research_stage' (which contains 'age')."""
    cols = list(columns)
    if token in cols:                                    # exact match wins
        return token
    cands = [c for c in cols if c not in _RESERVED
             and re.search(rf"(^|_){re.escape(token)}(_|$)", c.lower())]
    return cands[0] if cands else None


def build_structures(columns):
    """Group body_comp_<region>_* columns into region specialists (+ whole-body adipose)."""
    regions = sorted(["android", "gynoid", "arm_left", "arm_right", "arms",
                      "leg_left", "leg_right", "legs", "trunk", "total"],
                     key=len, reverse=True)
    groups = {}
    for c in columns:
        if c.startswith("body_comp_"):
            rest = c[len("body_comp_"):]
            reg = next((r for r in regions if rest.startswith(r)), "other")
            groups.setdefault(reg, []).append(c)
        elif c.startswith("total_scan_"):
            groups.setdefault("whole_body_adipose", []).append(c)
    return groups


def main(disease):
    os.makedirs(OUT_DIR, exist_ok=True)

    # X — body composition phenotypes
    bc = baseline(get_df("body_composition", "body_composition"))
    pheno_cols = [c for c in bc.columns if c.startswith(("body_comp_", "total_scan_"))]
    X = bc[pheno_cols]

    # C — confounders: age/sex + anthropometrics
    ag = baseline(get_df("curated_phenotypes", "age_sex"))
    age_col = pick_col(ag.columns, "age")
    sex_col = pick_col(ag.columns, "sex")
    C = ag[[c for c in (age_col, sex_col) if c]].rename(columns={age_col: "age", sex_col: "sex"})
    an = baseline(get_df("anthropometrics", "anthropometrics"))
    an_num = [c for c in ["height", "weight", "bmi", "waist_circumference",
                          "hip_circumference", "waist_to_hip_ratio"] if c in an.columns]
    C = C.join(an[an_num], how="outer")

    # D — label
    lab = baseline(get_df("curated_phenotypes", disease))
    lcol = f"{disease}__curated_phenotype"
    cats = sorted(lab[lcol].dropna().astype(str).unique())
    is_pos = POSITIVE[disease]
    lab[disease] = lab[lcol].astype(str).map(is_pos).astype("Int64")
    D = lab[[disease]]

    merged = X.join(C, how="inner").join(D, how="inner").dropna(subset=[disease])
    merged = merged.reset_index()  # participant_id becomes a column

    out_csv = os.path.join(OUT_DIR, "merged_data.csv")
    merged.to_csv(out_csv, index=False)

    structures = build_structures(pheno_cols)
    with open(os.path.join(OUT_DIR, "structures.json"), "w") as f:
        json.dump(structures, f, indent=2)

    pos = int(merged[disease].sum())
    print(f"label disease   : {disease}  (categories: {cats})")
    print(f"merged_data.csv : {merged.shape[0]} rows x {merged.shape[1]} cols -> {out_csv}")
    print(f"  phenotypes(X) : {len(pheno_cols)}   confounders(C): {list(C.columns)}")
    print(f"  label({disease}) positives: {pos}  rate: {pos/len(merged):.3f}")
    print(f"structures.json : {len(structures)} region groups -> {list(structures)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="diabetes", choices=list(POSITIVE))
    main(ap.parse_args().disease)
