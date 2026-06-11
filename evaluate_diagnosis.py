"""Diagnosis evaluation reproducing the MESHAgents protocol (paper §3 + Classification_methods.ipynb).

Per the paper text ("All results are obtained through five-fold cross-validation"):
  - two feature sets, BOTH with confounders appended:
       expert_selected + confounders     (clinically standard body-composition set)
       mesh_discovered + confounders      (consensus set from the agent PheWAS pipeline)
  - per disease label: balance classes by random undersampling (random_state=42, as the notebook),
    mean-impute, then STRATIFIED 5-FOLD CV; the three paper classifiers (AdaBoost, LDA, SVM);
    report AUC + Recall (mean +/- std across folds).
  - headline metric: (discovered - expert) AUC and Recall, echoing the paper's "-0.004 +/- 0.010".

The discovered set is read from results/discovered_phenotypes_*.json (written by main_bodycomp.py).
A RANDOM-K set is included as a sanity baseline.

Run from the repo root:  python evaluate_diagnosis.py
"""
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence LDA collinearity / convergence noise (cosmetic)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED = os.path.join(HERE, "MESHAgents", "data", "merged_data.csv")
RESULTS = os.path.join(HERE, "MESHAgents", "results")

DISEASE_NAMES = ["diabetes", "hypertension", "abdominal_adiposity", "mafld", "osteoporosis"]
CONFOUNDERS = ["age", "sex", "bmi", "weight", "waist_circumference"]
N_FOLDS = 5            # paper: five-fold cross-validation
BALANCE_SEED = 42     # class-balancing seed (matches the notebook)

# Clinically standard compact body-composition set (analogue of the cardiac expert_selected list).
EXPERT_BODYCOMP = [
    "total_scan_vat_mass", "total_scan_vat_volume", "total_scan_sat_mass",
    "body_comp_total_fat_mass", "body_comp_android_fat_mass", "body_comp_gynoid_fat_mass",
    "body_comp_trunk_fat_mass", "body_comp_total_lean_mass", "body_comp_total_fat_free_mass",
    "body_comp_android_region_percent_fat", "body_comp_total_region_percent_fat",
    "body_comp_legs_fat_mass", "body_comp_arms_fat_mass", "body_comp_total_bone_mass",
]


def latest_discovered():
    files = sorted(glob.glob(os.path.join(RESULTS, "discovered_phenotypes_*.json")))
    if not files:
        return []
    return json.load(open(files[-1])).get("discovered_phenotypes", [])


def classifiers():
    # the three classifiers used in the paper (AdaBoost, LDA, SVM)
    return {
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=0),
        "AdaBoost": lambda: AdaBoostClassifier(random_state=0),
    }


def eval_feature_set(df, label, features, clf_factory):
    """Balance classes (undersample) -> mean-impute -> stratified 5-fold CV -> (AUC, Recall)."""
    cols = [c for c in features if c in df.columns]
    data = df[df[label].isin([0, 1])].copy()
    counts = data[label].value_counts()
    m = int(min(counts.get(0, 0), counts.get(1, 0)))
    if m < N_FOLDS * 2:
        return None
    z = data[data[label] == 0].sample(n=m, random_state=BALANCE_SEED)
    o = data[data[label] == 1].sample(n=m, random_state=BALANCE_SEED)
    bal = pd.concat([z, o])
    X = SimpleImputer(strategy="mean").fit_transform(bal[cols])
    y = bal[label].astype(int).values
    pipe = make_pipeline(clf_factory())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BALANCE_SEED)
    cv = cross_validate(pipe, X, y, cv=skf, scoring={"auc": "roc_auc", "recall": "recall"})
    return {"auc_mean": float(cv["test_auc"].mean()), "auc_std": float(cv["test_auc"].std()),
            "recall_mean": float(cv["test_recall"].mean()), "recall_std": float(cv["test_recall"].std())}


def main():
    df = pd.read_csv(MERGED)
    pheno = [c for c in df.columns if c.startswith(("body_comp_", "total_scan_"))]
    labels = [d for d in DISEASE_NAMES if d in df.columns]
    if not labels:
        raise SystemExit("No disease label column found in merged_data.csv")

    discovered = latest_discovered()
    expert = [c for c in EXPERT_BODYCOMP if c in df.columns]
    conf = [c for c in CONFOUNDERS if c in df.columns]
    rng = np.random.default_rng(0)
    random_k = list(rng.choice(pheno, size=min(len(expert), len(pheno)), replace=False))

    if not discovered:
        print("WARNING: no discovered_phenotypes_*.json found. Run src/main_bodycomp.py first.")
    print(f"confounders        : {conf}")
    print(f"expert_selected    : {len(expert)} phenotypes")
    print(f"mesh_discovered    : {len(discovered)} phenotypes -> {discovered}")
    print(f"random_k baseline  : {len(random_k)} phenotypes")

    feature_sets = {
        "expert+conf": expert + conf,
        "discovered+conf": discovered + conf,
        "random+conf": random_k + conf,
    }
    clfs = classifiers()
    report = {"confounders": conf, "expert": expert, "discovered": discovered,
              "n_folds": N_FOLDS, "results": {}}

    for label in labels:
        print(f"\n=== label: {label} (positives={int(df[label].sum())}/{int(df[label].notna().sum())}) ===")
        report["results"][label] = {}
        for clf_name in clfs:
            print(f"\n  [{clf_name}]  {'feature set':18s}  {'AUC':>16s}  {'Recall':>16s}")
            print("  " + "-" * 56)
            row = {}
            for fs_name, feats in feature_sets.items():
                if fs_name == "discovered+conf" and not discovered:
                    continue
                r = eval_feature_set(df, label, feats, clfs[clf_name])
                if r is None:
                    continue
                row[fs_name] = r
                print(f"  {' ':9s}{fs_name:18s}  "
                      f"{r['auc_mean']:.3f}+/-{r['auc_std']:.3f}   "
                      f"{r['recall_mean']:.3f}+/-{r['recall_std']:.3f}")
            # headline: discovered - expert (the paper's metric)
            if "discovered+conf" in row and "expert+conf" in row:
                d_auc = row["discovered+conf"]["auc_mean"] - row["expert+conf"]["auc_mean"]
                d_rec = row["discovered+conf"]["recall_mean"] - row["expert+conf"]["recall_mean"]
                row["discovered_minus_expert"] = {"d_auc": d_auc, "d_recall": d_rec}
                print(f"  {' ':9s}{'-> discovered - expert':18s}  "
                      f"dAUC={d_auc:+.3f}            dRecall={d_rec:+.3f}")
            report["results"][label][clf_name] = row

    out = os.path.join(RESULTS, "diagnosis_eval.json")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
