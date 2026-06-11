"""Quick integrity check of the synthetic dataset (schema, dtypes, keys, overlap, load path)."""
import pandas as pd
from pheno_io import get_df

TABLES = ["body_composition", "anthropometrics", "age_sex",
          "diabetes", "hypertension", "abdominal_adiposity", "mafld", "osteoporosis"]

dfs = {t: get_df("curated_phenotypes" if t not in
                 ("body_composition", "anthropometrics") else t, t) for t in TABLES}

print("=== shapes & index (loaded via pheno_io.get_df) ===")
for t, df in dfs.items():
    print(f"{t:20s} {str(df.shape):>12}  index={df.index.names}")

print("\n=== key dtypes (body_composition) ===")
bc = dfs["body_composition"].reset_index()
for c in ["participant_id", "research_stage", "body_comp_android_fat_mass",
          "total_scan_vat_mass", "cohort"]:
    print(f"  {c:30s} {bc[c].dtype}")

print("\n=== label column types & categories ===")
for d in ["diabetes", "hypertension", "abdominal_adiposity", "mafld", "osteoporosis"]:
    col = f"{d}__curated_phenotype"
    s = dfs[d][col]
    print(f"  {col:42s} dtype={str(s.dtype):10s} cats={list(pd.unique(s.astype(str)))}")

print("\n=== research_stage values ===")
print(" ", sorted(bc['research_stage'].unique()))

print("\n=== participant overlap vs body_composition (reference) ===")
ref = set(dfs["body_composition"].reset_index()["participant_id"])
print(f"  reference unique participants = {len(ref)}")
for t, df in dfs.items():
    ids = set(df.reset_index()["participant_id"])
    inter = len(ids & ref)
    print(f"  {inter:5d} ({100*inter/len(ref):5.1f}% of ref)  uniq={len(ids):5d}  {t}")

print("\n=== baseline merge smoke test (body_comp X + age_sex + diabetes label) ===")
def baseline(df):
    d = df.reset_index()
    d = d[d["research_stage"] == "00_00_visit"].drop_duplicates("participant_id")
    return d.set_index("participant_id")
X = baseline(dfs["body_composition"]).filter(regex="^(body_comp_|total_scan_)")
C = baseline(dfs["age_sex"])[["age", "sex"]]
lab = baseline(dfs["diabetes"])[["diabetes__curated_phenotype"]]
merged = X.join(C, how="inner").join(lab, how="inner")
print(f"  merged: {merged.shape[0]} participants x {merged.shape[1]} cols (X+C+label)")
print(f"  no missing in keys/label: {merged['diabetes__curated_phenotype'].notna().all()}")
