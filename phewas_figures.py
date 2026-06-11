"""Reproduce the paper's Fig 1 artifacts for the discovered phenotype set.

Fig 1(a): overlap between expert-selected and MESHAgents-discovered phenotypes (+ confounders).
Fig 1(b): disease-association heatmap -- for every (discovered phenotype x disease) pair, the
          confounder-adjusted association (odds ratio + significance stars **p<0.01, *p<0.05),
          across all available disease labels, exactly mirroring Fig 1(b).

Outputs (MESHAgents/results/):
  fig1a_overlap.json
  fig1b_association_grid.csv        (odds ratio + p-value per phenotype x disease)
  fig1b_heatmap.png                 (odds-ratio heatmap with significance stars)

Run from the repo root:  python phewas_figures.py
"""
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

import build_merged_data as bmd       # reuse baseline(), POSITIVE, pick_col, get_df
from pheno_io import get_df

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "MESHAgents", "results")
DEMOGRAPHIC_CONFOUNDERS = ["age", "sex"]
EXPERT_BODYCOMP = [
    "total_scan_vat_mass", "total_scan_vat_volume", "total_scan_sat_mass",
    "body_comp_total_fat_mass", "body_comp_android_fat_mass", "body_comp_gynoid_fat_mass",
    "body_comp_trunk_fat_mass", "body_comp_total_lean_mass", "body_comp_total_fat_free_mass",
    "body_comp_android_region_percent_fat", "body_comp_total_region_percent_fat",
    "body_comp_legs_fat_mass", "body_comp_arms_fat_mass", "body_comp_total_bone_mass",
]


def latest_json(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return json.load(open(files[-1])) if files else None


def load_phenotypes_and_labels():
    """Phenotypes + demographics + ALL disease labels (binarized via build_merged_data.POSITIVE)."""
    bc = bmd.baseline(get_df("body_composition", "body_composition"))
    pheno_cols = [c for c in bc.columns if c.startswith(("body_comp_", "total_scan_"))]
    X = bc[pheno_cols]
    ag = bmd.baseline(get_df("curated_phenotypes", "age_sex"))
    age_col, sex_col = bmd.pick_col(ag.columns, "age"), bmd.pick_col(ag.columns, "sex")
    C = ag[[c for c in (age_col, sex_col) if c]].rename(columns={age_col: "age", sex_col: "sex"})
    labels = {}
    for d in bmd.POSITIVE:
        try:
            lab = bmd.baseline(get_df("curated_phenotypes", d))
        except Exception:
            continue
        lcol = f"{d}__curated_phenotype"
        if lcol in lab.columns:
            labels[d] = lab[lcol].astype(str).map(bmd.POSITIVE[d]).astype("float")
    frame = X.join(C, how="inner").join(pd.DataFrame(labels), how="inner")
    return frame, pheno_cols, list(labels)


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def assoc_or_p(df, phen, disease, confounders):
    """Confounder-adjusted logistic association -> (odds ratio, p) for the phenotype coefficient."""
    conf = [c for c in confounders if c in df.columns]
    d = pd.concat([pd.to_numeric(df[disease], errors="coerce").rename("y"),
                   _z(df[phen]).rename("pheno"),
                   pd.DataFrame({c: _z(df[c]) for c in conf})], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if d["y"].nunique() < 2 or len(d) < 30 or d["pheno"].std() == 0:
        return np.nan, np.nan
    try:
        res = sm.Logit(d["y"], sm.add_constant(d[["pheno"] + conf], has_constant="add")).fit(disp=0, maxiter=200)
        return float(np.exp(res.params["pheno"])), float(res.pvalues["pheno"])
    except Exception:
        return np.nan, np.nan


def stars(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def fig3_roc(frame, discovered, diseases, confounders):
    """Fig 3: LDA ROC curves per disease using discovered phenotypes + confounders (balanced 70/30)."""
    feats = [c for c in discovered + confounders if c in frame.columns]
    fig, ax = plt.subplots(figsize=(6, 6))
    for d in diseases:
        sub = frame[frame[d].isin([0, 1])].copy()
        m = int(min((sub[d] == 0).sum(), (sub[d] == 1).sum()))
        if m < 20:
            continue
        bal = pd.concat([sub[sub[d] == 0].sample(n=m, random_state=42),
                         sub[sub[d] == 1].sample(n=m, random_state=42)])
        X = SimpleImputer(strategy="mean").fit_transform(bal[feats])
        y = bal[d].astype(int).values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, prob)
        ax.plot(fpr, tpr, label=f"{d} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_title("Fig 3: ROC Curves — LDA (Model: MESH)\ndiscovered phenotypes + confounders",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", prop={"size": 9})
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS, "fig3_roc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


def main():
    disc = latest_json("discovered_phenotypes_*.json")
    if not disc:
        raise SystemExit("No discovered_phenotypes_*.json found; run src/main_bodycomp.py first.")
    discovered = disc["discovered_phenotypes"]
    frame, pheno_cols, diseases = load_phenotypes_and_labels()
    expert = [c for c in EXPERT_BODYCOMP if c in pheno_cols]

    # ---- Fig 1(a): overlap ----
    inter = sorted(set(discovered) & set(expert))
    overlap = {"discovered": discovered, "expert": expert, "overlap": inter,
               "jaccard": len(inter) / len(set(discovered) | set(expert)) if (discovered or expert) else 0.0}
    json.dump(overlap, open(os.path.join(RESULTS, "fig1a_overlap.json"), "w"), indent=2)
    print(f"Fig 1a: {len(inter)}/{len(expert)} expert phenotypes rediscovered "
          f"(Jaccard={overlap['jaccard']:.3f}): {inter}")

    # ---- Fig 1(b): discovered phenotype x disease association grid ----
    rows = []
    for phen in discovered:
        row = {"phenotype": phen}
        for d in diseases:
            orr, p = assoc_or_p(frame, phen, d, DEMOGRAPHIC_CONFOUNDERS)
            row[f"{d}_OR"] = round(orr, 3) if np.isfinite(orr) else np.nan
            row[f"{d}_p"] = p
        rows.append(row)
    grid = pd.DataFrame(rows).set_index("phenotype")
    grid.to_csv(os.path.join(RESULTS, "fig1b_association_grid.csv"))

    OR = grid[[f"{d}_OR" for d in diseases]].copy()
    OR.columns = diseases
    P = grid[[f"{d}_p" for d in diseases]].copy()
    P.columns = diseases
    nsig = int((P < 0.05).sum().sum())
    print(f"Fig 1b: {OR.shape[0]} discovered phenotypes x {len(diseases)} diseases; "
          f"{nsig} associations significant at p<0.05")

    fig, ax = plt.subplots(figsize=(1.4 * len(diseases) + 3, 0.45 * len(discovered) + 2))
    im = ax.imshow(OR.values, cmap="bwr", vmin=0.4, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(diseases))); ax.set_xticklabels(diseases, rotation=45, ha="right")
    ax.set_yticks(range(len(discovered))); ax.set_yticklabels(discovered, fontsize=7)
    for i in range(OR.shape[0]):
        for j in range(OR.shape[1]):
            s = stars(P.values[i, j]) if np.isfinite(P.values[i, j]) else ""
            if s:
                ax.text(j, i, s, ha="center", va="center", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Odds ratio")
    ax.set_title("Fig 1(b): discovered phenotype × disease associations\n(** p<0.01, * p<0.05)")
    plt.tight_layout()
    out_png = os.path.join(RESULTS, "fig1b_heatmap.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved -> {os.path.join(RESULTS, 'fig1a_overlap.json')}")
    print(f"saved -> {os.path.join(RESULTS, 'fig1b_association_grid.csv')}")
    print(f"saved -> {out_png}")

    # ---- Fig 3: ROC curves (LDA) using discovered features ----
    fig3_roc(frame, discovered, diseases, DEMOGRAPHIC_CONFOUNDERS)


if __name__ == "__main__":
    main()
