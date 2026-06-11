"""Generate synthetic HPP tables that mimic the real Pheno AI schema for local development.

Mirrors what we observed in the HPP TRE:
  - tables stored flat with participant_id (int64) + research_stage (string) keys (+ cohort, array_index)
  - body_composition: body_comp_<region>_<measure> doubles + body_comp_dose + total_scan_* doubles
  - anthropometrics: height/weight/bmi/circumferences
  - age_sex: the universal confounder table inside curated_phenotypes (age, sex)
  - disease tables: <disease>__curated_phenotype (multi-class categorical/dictionary) + sub-fields + self-report bools
  - multi-visit rows ("00_00_visit" baseline, "01_00_call" follow-up with mostly-NaN measures)
  - body_composition is the LIMITING cohort; disease/confounder tables cover a superset (mirrors real overlap)
  - a latent per-participant "risk" couples adiposity phenotypes with disease labels so the pilot finds real signal
"""
import os
import numpy as np
import pandas as pd

SYNTH_DIR = os.environ.get("SYNTH_DIR", "./synth_hpp")
N_POOL = 600                       # total participants
SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs(SYNTH_DIR, exist_ok=True)

PIDS = (1_000_000 + np.arange(N_POOL)).astype("int64")

# ---- latent causal structure with GENUINE confounding (so the PheWAS can discover confounders) ----
# DAG:   age, sex  ->  metabolic risk  ->  disease
#        age, sex  ->  adiposity latent (ADI)  ->  {fat phenotypes, weight/waist/bmi}
#        risk      ->  adiposity latent
# Hence age, sex, bmi, weight, waist are common causes of both phenotypes AND disease == true confounders.
_sex = rng.integers(0, 2, N_POOL)                          # 0/1
_age = rng.uniform(40, 70, N_POOL)
_zage = (_age - _age.mean()) / _age.std()
# metabolic risk driven by age + sex + an independent metabolic component
_risk_lin = 0.7 * _zage + 0.7 * (2 * _sex - 1) + rng.normal(0, 1.0, N_POOL)
_risk = (np.argsort(np.argsort(_risk_lin)) + 0.5) / N_POOL          # rank-normalised to (0,1)
# adiposity latent driven by risk + age + sex (standardised); loads onto fat phenotypes & anthropometrics
_adi_lin = 1.0 * (_risk - 0.5) * 2 + 0.5 * _zage + 0.4 * (2 * _sex - 1) + rng.normal(0, 0.6, N_POOL)
_adi = (_adi_lin - _adi_lin.mean()) / _adi_lin.std()

RISK = dict(zip(PIDS, _risk))                              # metabolic risk 0..1 (drives disease)
SEX = dict(zip(PIDS, _sex))                                # 0/1
AGE = dict(zip(PIDS, _age.round(1)))
ADI = dict(zip(PIDS, _adi))                                # standardised adiposity latent (confounded)
TZ = "asia/jerusalem"


def subset(frac):
    """A random participant subset of size frac*N_POOL (sorted)."""
    k = int(round(frac * N_POOL))
    return np.sort(rng.choice(PIDS, size=k, replace=False))


def skeleton(pids, followup_frac=0.4):
    """participant_id + research_stage rows; some participants get a 2nd (follow-up) visit."""
    rows = [(int(p), "00_00_visit") for p in pids]
    rows += [(int(p), "01_00_call") for p in pids if rng.random() < followup_frac]
    df = pd.DataFrame(rows, columns=["participant_id", "research_stage"])
    df["cohort"] = "10k"
    df["array_index"] = 0
    return df


def add_dates(df):
    base = np.datetime64("2021-01-01")
    df["collection_date"] = base + rng.integers(0, 700, len(df)).astype("timedelta64[D]")
    df["timezone"] = pd.Categorical([TZ] * len(df))
    df["collection_timestamp"] = np.nan          # matches real (double, mostly NaN)
    return df


def risk_of(df):
    return df["participant_id"].map(RISK).to_numpy()


def adi_of(df):
    return df["participant_id"].map(ADI).to_numpy()


def sex_of(df):
    return df["participant_id"].map(SEX).to_numpy()


def zage_of(df):
    a = df["participant_id"].map(AGE).to_numpy()
    return (a - 55.0) / 9.0


# ----------------------------------------------------------------------------- body_composition
REGIONS = ["android", "gynoid", "arm_left", "arm_right", "arms", "arms_diff",
           "leg_left", "leg_right", "legs", "legs_diff", "trunk", "trunk_left",
           "trunk_right", "trunk_diff", "total", "total_left", "total_right", "total_diff"]
MEASURES = ["bone_mass", "fat_free_mass", "fat_mass", "lean_mass",
            "region_percent_fat", "tissue_mass", "tissue_percent_fat", "total_mass"]


def make_body_composition():
    df = skeleton(subset(0.83))                  # limiting imaging cohort (~500)
    n = len(df)
    a = adi_of(df)                               # standardised adiposity latent (confounded by age/sex/risk)
    sx = sex_of(df)
    za = zage_of(df)
    for reg in REGIONS:
        for m in MEASURES:
            if "percent" in m:
                base = rng.normal(0.30, 0.07, n)
                if reg in ("android", "trunk", "total"):       # adiposity phenotypes load on ADI
                    base = base + 0.085 * a + 0.012 * za - 0.012 * (2 * sx - 1)
                df[f"body_comp_{reg}_{m}"] = base.round(4)
            else:
                base = rng.normal(5000, 1200, n)
                if "fat_mass" in m and reg in ("android", "trunk", "total"):
                    base = base * (1 + 0.42 * a + 0.06 * za)
                df[f"body_comp_{reg}_{m}"] = base.round(4)
    df["body_comp_dose"] = 0.4
    for c in ["sat_area", "sat_mass", "sat_volume"]:           # subcutaneous fat: moderate ADI loading
        df[f"total_scan_{c}"] = (rng.normal(1200, 350, n) * (1 + 0.22 * a)).round(3)
    for c in ["vat_area", "vat_mass", "vat_volume"]:           # visceral fat carries the strongest signal
        df[f"total_scan_{c}"] = (rng.normal(600, 200, n) * (1 + 0.55 * a + 0.08 * za)).round(3)
    df["total_scan_dose"] = 0.4
    df = add_dates(df)
    front = ["collection_date", "timezone", "collection_timestamp"]
    pheno = [c for c in df.columns if c.startswith(("body_comp_", "total_scan_"))]
    return df[front + pheno + ["participant_id", "cohort", "research_stage", "array_index"]]


# ----------------------------------------------------------------------------- anthropometrics
def make_anthropometrics():
    df = skeleton(subset(0.98))
    n = len(df)
    a = adi_of(df)                               # shared adiposity latent -> genuine confounding
    sx = sex_of(df)
    df = add_dates(df)
    df["height"] = (rng.normal(170, 9, n) + 6.0 * (2 * sx - 1)).round(1)     # taller for sex=1
    df["weight"] = (rng.normal(72, 12, n) + 9.0 * a + 4.0 * (2 * sx - 1)).round(1)
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(2)
    df["waist_circumference"] = (rng.normal(85, 10, n) + 7.0 * a).round(1)
    df["neck_circumference"] = rng.normal(37, 3, n).round(1)
    df["hip_circumference"] = (rng.normal(102, 9, n) + 3.0 * a).round(1)
    df["waist_to_hip_ratio"] = (df["waist_circumference"] / df["hip_circumference"]).round(3)
    cols = ["collection_date", "collection_timestamp", "height", "weight", "bmi",
            "waist_circumference", "neck_circumference", "hip_circumference",
            "waist_to_hip_ratio", "timezone", "participant_id", "cohort",
            "research_stage", "array_index"]
    return df[cols]


# ----------------------------------------------------------------------------- age_sex (confounders)
def make_age_sex():
    df = skeleton(subset(1.0), followup_frac=0.0)
    df["age"] = df["participant_id"].map(AGE)
    df["sex"] = df["participant_id"].map(SEX).astype("int64")
    return df[["age", "sex", "participant_id", "cohort", "research_stage", "array_index"]]


# ----------------------------------------------------------------------------- disease label tables
# category lists ordered control -> strongest case; sampling is risk-weighted.
DISEASES = {
    "diabetes": {
        "cats": ["non-diabetes", "prediabetes", "diabetes", "diabetes_medication"],
        "subfields": {"diabetes__bloodtest_hba1c": (5.4, 0.6, 2.5),
                      "diabetes__bloodtest_fpg": (95, 12, 30),
                      "diabetes__cgm_mage": (40, 10, 20)},
        "bools": ["diabetes__self_reported_diabetic", "diabetes__self_reported_prediabetic"],
    },
    "hypertension": {
        "cats": ["Non hypertensive", "High BP without diagnosis of hypertension",
                 "Suspected hypertension", "Hypertensive"],
        "subfields": {"hypertension__sitting_blood_pressure_systolic": (120, 12, 25),
                      "hypertension__sitting_blood_pressure_diastolic": (78, 8, 12)},
        "bools": ["hypertension__self_reported_hypertension"],
    },
    "abdominal_adiposity": {
        "cats": ["normal abdominal adiposity", "intermediate abdominal adiposity risk",
                 "high abdominal adiposity risk"],
        "subfields": {"abdominal_adiposity__waist_circumference": (88, 11, 18),
                      "abdominal_adiposity__vat_index": (25, 10, 25)},
        "bools": [],
    },
    "mafld": {
        "cats": ["Normal liver", "Suspected MAFLD", "MAFLD"],
        "subfields": {"mafld__bloodtest_alt": (24, 8, 18),
                      "mafld__bloodtest_ast": (22, 6, 12),
                      "mafld__fibrosis_4": (1.1, 0.4, 0.6)},
        "bools": ["mafld__self_reporting"],
    },
    "osteoporosis": {
        "cats": ["normal BMD", "osteopenia", "osteoporosis"],
        "subfields": {"osteoporosis__femur_bmd_min": (1.0, 0.12, -0.15),
                      "osteoporosis__spine_l1_l4_bmd": (1.1, 0.13, -0.15)},
        "bools": ["osteoporosis__self_reported_osteoporosis",
                  "osteoporosis__self_reported_osteopenia"],
        "embeds_age_sex": True,
    },
}


def sample_category(cats, r):
    """Risk-weighted multinomial over ordered categories (higher risk -> later categories)."""
    k = len(cats)
    out = []
    for ri in r:
        w = np.array([np.exp(-4.5 * abs(i / (k - 1) - ri)) for i in range(k)])
        out.append(rng.choice(k, p=w / w.sum()))
    return np.array(out)


def make_disease(name, spec):
    df = skeleton(subset(rng.uniform(0.95, 1.0)))     # diseases cover ~all participants
    n = len(df)
    r = risk_of(df)
    idx = sample_category(spec["cats"], r)
    df[f"{name}__curated_phenotype"] = pd.Categorical([spec["cats"][i] for i in idx],
                                                      categories=spec["cats"])
    is_base = (df["research_stage"] == "00_00_visit").to_numpy()
    for col, (lo, sd, slope) in spec["subfields"].items():
        vals = rng.normal(lo, sd, n) + slope * (idx / (len(spec["cats"]) - 1))
        vals = np.where(is_base, vals.round(3), np.nan)     # follow-up mostly NaN, like real
        df[col] = vals
    for b in spec["bools"]:
        flags = (rng.random(n) < (0.05 + 0.25 * (idx / (len(spec["cats"]) - 1))))
        df[b] = pd.array(np.where(is_base, flags, pd.NA), dtype="boolean")
    if spec.get("embeds_age_sex"):
        df[f"{name}__sex"] = pd.Categorical(df["participant_id"].map(SEX).astype(str))
        df[f"{name}__age_at_research_stage"] = df["participant_id"].map(AGE)
    label_col = f"{name}__curated_phenotype"
    other = [c for c in df.columns if c not in
             ("participant_id", "research_stage", "cohort", "array_index", label_col)]
    return df[[label_col] + other + ["participant_id", "research_stage", "cohort", "array_index"]]


def main():
    tables = {
        "body_composition": make_body_composition(),
        "anthropometrics": make_anthropometrics(),
        "age_sex": make_age_sex(),
    }
    for name, spec in DISEASES.items():
        tables[name] = make_disease(name, spec)

    manifest = []
    for name, df in tables.items():
        path = os.path.join(SYNTH_DIR, f"{name}.parquet")
        df.to_parquet(path, index=False)
        npart = df["participant_id"].nunique()
        manifest.append((name, df.shape[0], df.shape[1], npart))
        print(f"wrote {name:20s} rows={df.shape[0]:5d} cols={df.shape[1]:4d} participants={npart}")

    man = pd.DataFrame(manifest, columns=["table", "rows", "cols", "unique_participants"])
    man.to_csv(os.path.join(SYNTH_DIR, "_manifest.csv"), index=False)
    print(f"\nSYNTH_DIR = {os.path.abspath(SYNTH_DIR)}")


if __name__ == "__main__":
    main()
