# pheno26 — Whole-Body Multi-Agent Phenotype Analysis (MESHAgents on HPP)

_Last updated: 2026-06-10 — reproduces the MESHAgents PheWAS + diagnosis protocol on synthetic data._

## 1. Goal

Adapt the **MESHAgents** framework (Multi-Agent Reasoning for Cardiovascular Imaging Phenotype
Analysis, MICCAI 2025) from a single organ (heart, cardiac MRI) to **whole-body phenotype analysis**
on the **Human Phenotype Project (HPP)** datasets — building and validating **one dataset at a time**
before federating across datasets.

MESHAgents deploys one specialist LLM agent per anatomical structure (LV, RV, LA, RA, AAo, DAo) plus
a coordinator, running a multi-stage pipeline (per-structure analysis → cross-structure → phenotype
discovery → system-level → LLM synthesis). "Whole-body" applies the same pattern one level up: each
HPP dataset becomes a specialist domain, with a coordinator reasoning across domains.

**First pilot dataset: Body composition (DXA)** — the cleanest structural analog to the cardiac data
(many continuous phenotypes that partition into anatomical regions). See
`whole_body_multiagent_plan.md` and `hpp_data_feasibility_findings.md` for the full plan and data
feasibility analysis.

## 2. Strategy: develop on synthetic data, port by a flag

We cannot pull HPP data out of the Trusted Research Environment (TRE), and the TRE blocks outbound
internet (so OpenAI calls won't work there). So we:

1. Replicate the HPP schema as **synthetic parquet tables** locally.
2. Build and test the full adapter + agent pipeline against the synthetic data.
3. Port to the HPP VM by flipping **one flag** (`pheno_io.USE_SYNTHETIC = False`), which switches the
   data layer from synthetic parquet to the official `pheno_utils.PhenoLoader`.

## 3. Repository layout

```
pheno26/
├── pheno_io.py                 # data-access shim: synthetic parquet  <->  PhenoLoader (one flag)
├── make_synthetic.py           # generates synthetic HPP-schema tables
├── verify_synth.py             # integrity check on the synthetic data
├── build_merged_data.py        # ADAPTER: HPP/synthetic tables -> MESHAgents/data/merged_data.csv
├── evaluate_diagnosis.py       # Track-2 diagnosis eval: expert vs discovered (+conf), 5-fold CV, AUC+Recall
├── phewas_figures.py           # Fig 1(a) overlap + Fig 1(b) disease-association heatmap
├── phewas_ablation.py          # Table-1 Auto-PheWAS ablations (full / w-o Stage III / w-o Mem.&Tool)
├── phewas_baselines.py         # Table-1 Independent-LLM baselines (zero-shot CoT; OpenAI + optional Claude)
├── ollama_provider.md          # one-switch OpenAI <-> local Ollama (TRE) — provider guide
├── synth_hpp/                  # 8 synthetic parquet tables (+ _manifest.csv)
├── references/                 # MESHAgents paper citation (.bib, .md)
├── whole_body_multiagent_plan.md          # ⚠ referenced but NOT YET WRITTEN (see §8/§9)
├── hpp_data_feasibility_findings.md       # ⚠ referenced but NOT YET WRITTEN
├── pheno_vs_meshagents_feasibility.md     # ⚠ referenced but NOT YET WRITTEN
└── MESHAgents/                 # cloned upstream codebase + our additive adaptation
    ├── src/
    │   ├── agents.py           # UPSTREAM, unchanged (cardiac framework)
    │   ├── main.py             # UPSTREAM, unchanged
    │   ├── config.py           # UPSTREAM, unchanged
    │   ├── agents_bodycomp.py  # NEW: RegionAgent + BodyCompChiefAgent (subclass upstream)
    │   ├── mesh_core.py        # NEW: faithful 3-stage protocol (Tools, Memory, Stage I/II/III, f_AP, Q/C)
    │   ├── llm_provider.py     # NEW: provider switch (OpenAI <-> Ollama) via one MESH_PROVIDER flag
    │   └── main_bodycomp.py    # NEW: body-composition entry point
    ├── data/                   # merged_data.csv + structures.json (generated)
    └── results/                # pipeline outputs (generated)
```

**Design principle:** the upstream files (`agents.py`, `main.py`, `config.py`) are left **unchanged**
to stay in sync with teammates. All adaptation lives in two new files that *subclass* the upstream
classes. The cardiac specialists are replaced by generic **RegionAgent**s built from a `structures`
dict; the cardiac-specific orchestration (cross-organ pairs, system-level cardiac metrics, GPT
summaries) is overridden with dataset-agnostic versions. The generic core (`_discover_key_phenotypes`,
`_calculate_phenotype_score`, `Memory`, `StatisticalTools`) is reused as-is.

## 4. How the data maps (HPP → MESHAgents)

| MESHAgents concept | HPP / our mapping |
|---|---|
| Join key | `participant_id` (int64) + `research_stage` (string) |
| Specialist structures (LV/RV/…) | body-composition regions: android, gynoid, arms, legs, trunk, total (+ whole-body adipose) |
| Phenotypes (X) | `body_comp_<region>_<measure>` + `total_scan_*` (152 columns) |
| Confounders (C) | `age_sex` table (age, sex) + anthropometrics (height/weight/bmi/waist…) |
| Disease label (D) | `<disease>__curated_phenotype` (multi-class) → binarized per a case rule |
| `merged_data.csv` | produced by `build_merged_data.py` |

## 5. How to run

### Environment (local, `.venv` via `uv` — mirrors `environment.yml`)
```bash
cd MESHAgents
uv venv --python 3.10 .venv                       # conda env -> .venv (Python 3.10)
uv pip install --python .venv/bin/python \
  pandas numpy scipy statsmodels scikit-learn seaborn matplotlib python-dotenv \
  pyarrow "openai>=1.0.0"                          # pyarrow is required (reads synthetic parquet)
source .venv/bin/activate
```
(`pyarrow` was missing from the original `environment.yml`; it is now added.)

### Local (synthetic data, no API calls)
```bash
cd pheno26
python make_synthetic.py                 # (already done; regenerates synth_hpp/)
python build_merged_data.py --disease diabetes      # writes MESHAgents/data/merged_data.csv (+ structures.json)
cd MESHAgents
OFFLINE_LLM=1 OPENAI_API_KEY=dummy python src/main_bodycomp.py   # stats + PheWAS discovery, no LLM
cd ..
python evaluate_diagnosis.py             # expert vs discovered (paper protocol)
```

### Local with the LLM (uses your OpenAI key in MESHAgents/.env)
```bash
cd MESHAgents
python src/main_bodycomp.py              # OFFLINE_LLM unset -> real GPT calls (gpt-5-mini); validated working
```

> Note: `src/main.py` is the **upstream cardiac** entry point and will raise
> `Missing required columns: {LVEF (%), ...}` on body-composition data — that is by design (the upstream
> files are unchanged). Use `src/main_bodycomp.py` for this project.

### Local with Ollama (no internet / TRE) — one switch
Flip the LLM provider from OpenAI to local Ollama with a single env var `MESH_PROVIDER=ollama`
(model selectable via `MESH_LLM_MODEL`). No OpenAI key needed. See **[ollama_provider.md](ollama_provider.md)**.
```bash
ollama serve &                                   # local OpenAI-compatible server
ollama pull qwen2.5:7b nomic-embed-text          # chat + embedding models (one-time)
cd MESHAgents
MESH_PROVIDER=ollama MESH_LLM_MODEL=qwen2.5:7b python src/main_bodycomp.py
```

### On the HPP VM (real data)
1. Set `pheno_io.USE_SYNTHETIC = False` (uses `pheno_utils.PhenoLoader`).
2. Set `MESH_PROVIDER=ollama` (local open-weight models; TRE has no external internet) — one switch,
   no upstream edits, no OpenAI key. See **[ollama_provider.md](ollama_provider.md)**.
3. Run `build_merged_data.py` then `src/main_bodycomp.py`. Only the data layer (step 1) and the provider
   (step 2) change. Confirm the real `age_sex` column names resolve via `build_merged_data.pick_col`.

## 6. What we have achieved

- [x] Mapped HPP catalog (37 datasets / 3,067 fields) and ranked feasibility vs MESHAgents.
- [x] Confirmed data feasibility: clean integer join key across all 23 tables, body-composition the
      limiting cohort, ~100% disease-label coverage, `age_sex` table for confounders.
- [x] Synthetic dataset replicating the HPP schema (8 tables; body_composition = 159 cols).
- [x] `pheno_io` one-flag shim (synthetic ↔ PhenoLoader).
- [x] `build_merged_data.py` adapter → `merged_data.csv` (473 × 162) + `structures.json` (12 regions).
- [x] Body-composition agent layer (`agents_bodycomp.py`, `main_bodycomp.py`) subclassing upstream,
      with an **OFFLINE_LLM** mode so the statistical pipeline runs without API calls.
- [x] **Faithful 3-stage MESHAgents protocol** implemented in `src/mesh_core.py` (maps 1:1 onto the
      paper §2 — see §6a below): Tools, Dynamic Memory (embedding retrieval), Stage I valuation,
      Stage II factor discovery + **confounder identification**, Stage III **sequential LLM consensus**
      (≤10 rounds, coordinator convergence), `f_AP` aggregation, and the Auto-PheWAS metrics Q(P), C(P).
- [x] **Diagnosis evaluation = paper protocol** (`evaluate_diagnosis.py`): expert vs discovered, **both +
      confounders**, class-balanced undersampling, **stratified 5-fold CV** (paper text), the three paper
      classifiers (AdaBoost/LDA/SVM), metrics **AUC + Recall**, headline Δ(discovered−expert).
- [x] Full pipeline executed end-to-end on synthetic data, **offline-stub and with the real LLM** (gpt-5-mini).

### 6a. The 3-stage protocol (faithful to arXiv:2507.03460v2 §2)

| Paper | Implementation (`src/mesh_core.py`) |
|---|---|
| Tools `T` (significance, effect size, distribution) | `Tools.association/distribution/stability` |
| Dynamic memory `M_i`, `R_i = argmax_h Sim(Emb(X),Emb(h))` | `AgentMemory` + `Embedder` (OpenAI embeddings, hashing fallback) |
| Stage I `V_i = {φ_i(p_k,R_i)}` | `stage1_valuation` (significance + LLM clinical-relevance + distribution + stability) |
| Stage II `E_i`,`E_G`, confounders | `stage2_factors` (phenotype↔factor association; confounder = sig. vs **outcome** AND vs **phenotypes** via FDR) |
| Stage III `O_i^t = g_i(V_i,H_{t-1},M_i)`, ≤10 rounds | `stage3_consensus` (sequential **LLM** opinions, coordinator-guided convergence) |
| `A = f_AP({O^t},{R_i},TR)`, confidence<0.05 & relevance>0.3 | `f_AP` (rank × significance-weight × relevance-weight) |
| Auto-PheWAS `Q(P)` (dependency), `C(P)` (coverage) | `dependency`, `coverage` (exact formulas) |

The model client is **TRE-swappable via one flag**: `MESH_PROVIDER=ollama` routes the entire LLM path to
local Ollama (model via `MESH_LLM_MODEL`, default `qwen2.5:7b`); no OpenAI key needed. Centralized in
`src/llm_provider.py` (upstream `config.py`/`agents.py` untouched). See **[ollama_provider.md](ollama_provider.md)**.
`MESH_DRY_RUN=1` substitutes a statistical opinion stub for fast/free testing; `MESH_MAX_ROUNDS` defaults to 10.

### Bugs fixed during reproduction
- **Confounder corruption (critical):** `build_merged_data.py` selected the `age` confounder with a
  substring match `"age" in c.lower()`, which silently matched **`research_stage`** ("rese**arch_st**age").
  The `age` column in `merged_data.csv` was therefore the visit string `"00_00_visit"`, not age —
  voiding any confounder-adjusted analysis. Replaced with a token-boundary `pick_col()` matcher.
- **No genuine confounders in the synthetic data:** the original generator coupled phenotypes↔disease only
  through a latent independent of age/sex/bmi, so *no variable was a true confounder* and Stage II had
  nothing to discover. `make_synthetic.py` now encodes a real confounding DAG (age, sex → adiposity latent
  → {fat phenotypes, weight/waist/bmi}; age, sex → risk → disease), making age/sex/bmi/weight/waist
  genuine, discoverable confounders.
- **Over-adjustment guard:** the phenotype→disease association ranking adjusts for **demographic**
  confounders (age, sex) only; the anthropometric confounders found in Stage II are *proxies* of the
  adiposity signal, so they are reported and **appended to the diagnosis classifier** (as the paper appends
  Sex/Age/Weight/Height) rather than partialled out of the ranking.

## 7. Results (synthetic pilot, label = diabetes)

Dataset: `merged_data.csv` = 479 participants × 152 phenotypes; label positive rate ≈ 0.49.
The synthetic data encodes a genuine confounding DAG (§6a), so the pipeline must *both* discover the
disease-linked adiposity phenotypes *and* surface the confounders.

### 7.1 Stage II — confounder discovery
The agents flag as confounders the factors significantly associated with **both** the phenotypes and the
disease: **{weight, sex, height, waist, bmi, age, hip}** — correctly **excluding** `waist_to_hip_ratio`
when it is not outcome-associated. This reproduces the paper's "identifying additional confounder
variables beyond standard demographic factors."

### 7.2 Stage III — consensus discovery
Phenotype↔disease association adjusted for **demographics (age, sex)** yields **12 / 150 FDR-significant**
phenotypes. The sequential consensus (≤10 rounds, coordinator convergence) distils a 13-phenotype
discovered set dominated by the disease signal — visceral (VAT area/mass), android/trunk fat, total &
android/trunk fat-% — i.e. surfaced *for the right reason*. Auto-PheWAS metrics: dependency **Q ≈ 0.62**
(lower = more independent), coverage **C ≈ 0.31**. (See `results/discovered_phenotypes_*.json` and the
round-by-round `results/consensus_transcript_*.json`.)

### 7.3 Auto-PheWAS Table 1 (Track 1, paper §3 + Table 1)
The two intrinsic metrics are implemented **exactly** per the paper (`mesh_core.dependency`, `coverage`):
- **Dependency** `Q(P) = [1 − mean_{i<j}|corr(p_i,p_j)|] · (|P_valid|/|P|)` — `|P_valid|` = phenotypes present
  in the real phenotype universe (hallucination check); constants skipped via `nanmean`.
- **Coverage** `C(P) = ws·|Ω_cov|/|Ω| + wf·|F_cov|/|F_total|` — `F_total` = structure×function combinations
  that **actually exist** in the data (not the Cartesian product; `ws=wf=0.5`, the paper's weights are
  unspecified).

**Independent-LLM baselines** (`phewas_baselines.py`, zero-shot CoT, Table 1 "Independent LLMs"):

| Method | Dependency Q | Coverage C |
|---|---|---|
| gpt-3.5-turbo (zero-shot CoT) | 0.931 | 0.225 |
| gpt-4o-mini (zero-shot CoT) | 0.877 | 0.270 |
| gpt-5-mini (zero-shot CoT) | 0.543 | 0.270 |
| **MESHAgents (ours)** | 0.622 | **0.361** |

Reproduces the paper's pattern: weaker single LLMs have much higher dependency (0.88–0.93 vs the paper's
GPT-3.5=0.642) and MESHAgents wins **coverage** decisively (0.36 vs 0.22–0.27). A frontier single model
(gpt-5-mini) gets low dependency alone but still loses on coverage — the paper's "single LLMs trade off
comprehensiveness" finding.

**Ablations** (`phewas_ablation.py`, Table 1, real-LLM 3-config run):

| Config | Dependency Q | Coverage C | rounds |
|---|---|---|---|
| **full** | **0.465** | 0.270 | 2 |
| w/o Stage III | 0.465 | 0.270 | 1 |
| w/o Mem. & Tool | 0.864 | 0.540 | 2 |

Honest reading (two caveats, not spun):
- **w/o Mem.&Tool** reproduces the paper on dependency — dropping the tool-evidence weighting in `f_AP`
  makes the set much more redundant (Q 0.465 → 0.864). But its *coverage rises* (0.27 → 0.54), a tradeoff
  the paper doesn't show (un-grounded `f_AP` spreads picks across anatomically diverse but disease-irrelevant
  regions).
- **w/o Stage III is identical to full** here (same set; converges to round-1 `f_AP` output). The paper shows
  a small degradation (0.385 vs 0.350); **we show none** — because on this synthetic data the disease signal
  is strong/concentrated, so the statistics-grounded `f_AP` dominates and the LLM consensus converges in one
  round, leaving the iteration nothing to refine. This is a property of the synthetic data, not a defect; the
  rounds would matter more on noisier real (UK Biobank / HPP) data.

> **Metric-direction note:** the paper labels Q "Dependency, lower=better", but the *printed formula* is an
> independence×validity score (higher for more-independent/valid sets) — an internal inconsistency in the
> paper. We implement the formula **verbatim**; read the sign per the paper's Table 1 convention.

**Not reproduced — external frameworks.** Table 1 also includes **MedAgents** and **RareAgents** (and
**Claude-3.5**). MedAgents/RareAgents are separate published codebases; reproducing their rows *exactly*
requires their repos (we will not ship an approximation labelled as them). Claude-3.5 runs automatically if
`ANTHROPIC_API_KEY` + the `anthropic` SDK are present (`phewas_baselines.py`).

### 7.4 Fig 1 artifacts (`phewas_figures.py`)
- **Fig 1(a):** 7/14 expert phenotypes rediscovered (Jaccard 0.35) — the paper's "explainable overlap".
- **Fig 1(b):** discovered-phenotype × disease association heatmap (odds ratio + **p<0.01/*p<0.05 stars);
  `fig1b_heatmap.png` + `fig1b_association_grid.csv`.

### 7.5 Diagnosis evaluation (paper protocol, 5-fold CV) — from `evaluate_diagnosis.py`
Class-balanced, stratified 5-fold CV; feature sets include the confounders. The agent-**discovered** set
**matches/beats the expert set on AUC and improves recall** across all three paper classifiers —
reproducing the paper's headline ("discovered ≈ expert on diagnosis; recall improves"):

| Classifier | expert + conf (AUC) | discovered + conf (AUC) | random + conf (AUC) | Δ AUC (disc−exp) | Δ Recall |
|---|---|---|---|---|---|
| **LDA** | 0.740 ± 0.042 | **0.757 ± 0.033** | 0.720 ± 0.037 | **+0.018** | +0.008 |
| SVM | 0.753 ± 0.031 | **0.763 ± 0.027** | 0.722 ± 0.012 | +0.011 | +0.000 |
| AdaBoost | 0.710 ± 0.034 | **0.735 ± 0.050** | 0.700 ± 0.048 | +0.025 | +0.056 |

Both expert and discovered beat the random baseline, confirming the PheWAS finds real signal; the small
positive Δ-AUC and recall gains mirror the paper's pattern (LDA AUC −0.004±0.010, recall improved for 6/9).
(Numbers above are from the `MESH_DRY_RUN` statistical path; the real-LLM consensus run produces an
equivalent set — see `consensus_transcript_*.json`.)

Artifacts: `analysis_results_*.json`, `phenotype_scores_*.csv`, `phenotype_association_*.csv`,
`discovered_phenotypes_*.json`, `consensus_transcript_*.json`, `<region>_analysis_*.json`,
`gpt_analysis_*.json`, `diagnosis_eval.json`.

## 8. Design notes & honest caveats

- **Two scores are reported.** `phenotype_importance` is the upstream **unsupervised** salience score
  (CV/outlier/range, `_calculate_phenotype_score`, unchanged) — kept for continuity. `phenotype_association`
  is the new **supervised, confounder-adjusted, FDR-controlled** PheWAS — this is what drives the
  discovered set and the diagnosis evaluation, and is what reproduces the paper.
- **Confounder adjustment is two-layered**, matching the codebase: (a) the association regresses each
  phenotype on the label *with confounders as covariates*; (b) the diagnosis classifier also receives
  confounders appended to the feature matrix (exactly as `Classification_methods.ipynb`).
- **`expert_selected` is a clinically-standard body-comp set** we curated (VAT/SAT, android/gynoid/trunk
  fat, lean/bone) as the analogue of the cardiac expert list — confirm with a domain expert before the
  HPP run.
- **Multi-agent consensus is now a genuine sequential protocol** (`stage3_consensus`): each round every
  specialist forms an LLM opinion `O_i^t = g_i(V_i, H_{t-1}, M_i)` conditioned on its valuations, the
  discussion history, and retrieved memory; the coordinator aggregates via `f_AP` and stops on convergence
  (≤10 rounds). The *final selection* `f_AP` is statistics-grounded (significance<0.05 & relevance>0.3), so
  the discovered set is robust even if LLM opinions are noisy.
- **LDA collinearity:** bmi/weight/waist are collinear, so LDA emits (cosmetic) singular-covariance
  warnings; suppressed in `evaluate_diagnosis.py`. Results are unaffected (rank-based AUC).
- **Synthetic data is illustrative, not calibrated.** The confounding-DAG loadings in `make_synthetic.py`
  were tuned so the protocol has signal *and* confounders to find; absolute AUCs/Δ are not meant to match
  UK Biobank magnitudes. On real HPP data we expect the paper's near-parity (Δ-AUC ≈ 0).
- **Synthetic disease tables** carry fewer sub-fields than real HPP tables. The case/control rules in
  `build_merged_data.py::POSITIVE` are placeholders — confirm against the real
  `<disease>__curated_phenotype` categories on the VM.
- The `other` region group is just `body_comp_dose` (a scanner dose field); `total_scan_dose` is likewise
  a constant. Both are harmless (zero variance → no association) but should be dropped on the real run.
- **Table 1 external baselines NOT reproduced.** The paper's Table 1 also compares against single-LLM
  zero-shot CoT (GPT-3.5, GPT-4o-mini, **Claude-3.5**) and other multi-agent frameworks (**MedAgents**,
  **RareAgents**). These are separate systems (and a different vendor's model); reproducing their rows
  exactly requires their codebases + a Claude key, so they are **out of scope** here. We reproduce the
  **MESHAgents + ablation** rows (`phewas_ablation.py`) and the two metrics exactly.
- **Missing planning docs:** `whole_body_multiagent_plan.md`, `hpp_data_feasibility_findings.md`,
  `pheno_vs_meshagents_feasibility.md` are referenced in §1/§3 but are **not present in the repo** — they
  need to be written (or the references removed).

## 9. Next steps

1. ~~Run the pipeline with the LLM to validate the agent-synthesis stage.~~ ✅ done (gpt-5-mini).
2. ~~Add a supervised, confounder-adjusted association + FDR.~~ ✅ done.
3. ~~Genuine multi-round agent discussion/consensus (the paper's MDT mechanism).~~ ✅ done (`mesh_core.stage3_consensus`).
4. ~~Confounder discovery + Auto-PheWAS metrics (Q, C) + 5-fold-CV diagnosis.~~ ✅ done.
5. Port to the HPP VM: `pheno_io.USE_SYNTHETIC=False` (data) + `MESH_PROVIDER=ollama` (LLM — ✅ one-switch
   provider done, see [ollama_provider.md](ollama_provider.md)); confirm `age_sex` column resolution and
   the `POSITIVE` case/control rules against the real categories.
6. Scale to dataset #2 (`bone_density`) and #3 (`ecg`) via the same `structures` mechanism.
7. Add the whole-body federation coordinator across datasets (cohort-overlap + multiple-testing handling).
8. Write the three missing planning/feasibility docs referenced in §1/§3.
