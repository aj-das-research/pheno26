# RUNNING.md — step-by-step guide

How to clone, set up, point the code at data (synthetic **or** real Pheno datasets), and run the full
MESHAgents whole-body PheWAS pipeline. For the design/results see
[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md); for the LLM provider switch see
[ollama_provider.md](ollama_provider.md).

There are exactly **two switches** you set, independently:

| Switch | Where | Options |
|---|---|---|
| **Data source** | `pheno_io.py` → `USE_SYNTHETIC` | `True` = local synthetic parquet · `False` = real Pheno via `pheno_utils.PhenoLoader` |
| **LLM provider** | env `MESH_PROVIDER` | `openai` (default) · `ollama` (local, no internet) |

---

## 0. Prerequisites
- **Python 3.10** (the pinned version).
- [`uv`](https://docs.astral.sh/uv/) (recommended; `setup.sh` falls back to `venv`+`pip`).
- For local LLM / the TRE VM: **[Ollama](https://ollama.com)**.
- For real data: the **`pheno_utils`** package + access to the Pheno dataset directory (on the VM).

---

## 1. Clone & set up the environment

```bash
git clone https://github.com/aj-das-research/pheno26.git
cd pheno26

./setup.sh                      # creates ./.venv (Python 3.10), installs requirements, seeds .env
source .venv/bin/activate
```

`setup.sh` copies `.env.example` → `.env`. Open `.env` and configure the provider (next step).

---

## 2. Choose the LLM provider

### Option A — OpenAI (local dev, needs internet + key)
In `.env`:
```
OPENAI_API_KEY=sk-...your key...
# MESH_PROVIDER stays unset/openai
```

### Option B — Ollama (local models, no internet — the Pheno TRE VM)
```bash
ollama serve &                                   # start the local server
ollama pull qwen2.5:7b nomic-embed-text          # chat + embedding models (one-time)
```
In `.env` (no OpenAI key needed):
```
MESH_PROVIDER=ollama
MESH_LLM_MODEL=qwen2.5:7b          # any pulled Ollama tag
MESH_EMBED_MODEL=nomic-embed-text
# OLLAMA_HOST=http://localhost:11434   # only if Ollama runs elsewhere
```
Full details + model options: [ollama_provider.md](ollama_provider.md).

> Tip: `MESH_DRY_RUN=1` runs the whole statistical pipeline with **no LLM calls** (free, no key, no server) —
> good for a first smoke test or CI.

---

## 3. Choose the data source

### Option A — Synthetic data (default; runs anywhere, no PHI)
```bash
python make_synthetic.py                       # writes synth_hpp/*.parquet  (uses $SYNTH_DIR, default ./synth_hpp)
python build_merged_data.py --disease diabetes # -> MESHAgents/data/merged_data.csv (+ structures.json)
```
Disease choices: `diabetes | hypertension | abdominal_adiposity | mafld | osteoporosis`.

### Option B — Real Pheno datasets (on the VM)
The data layer is a one-flag shim in [`pheno_io.py`](pheno_io.py): set `USE_SYNTHETIC = False` and it loads
via the official `pheno_utils.PhenoLoader` instead of synthetic parquet.

1. **Install + configure `pheno_utils`** so it points at your Pheno dataset directory. `PhenoLoader(dataset)`
   reads from the location configured by `pheno_utils` (its standard config / `PHENO_*` settings on the VM).
   Confirm a dataset loads in a Python shell:
   ```python
   from pheno_utils import PhenoLoader
   pl = PhenoLoader("body_composition"); print(pl.dfs.keys())
   ```
2. **Flip the flag** in `pheno_io.py`:
   ```python
   USE_SYNTHETIC = False
   ```
3. **Confirm dataset/table names** match your Pheno catalog. `build_merged_data.py` requests:
   - phenotypes: `get_df("body_composition", "body_composition")`
   - confounders: `get_df("curated_phenotypes", "age_sex")` + `get_df("anthropometrics", "anthropometrics")`
   - disease label: `get_df("curated_phenotypes", "<disease>")` → column `<disease>__curated_phenotype`
   - baseline visit filter: `STAGE = "00_00_visit"`
   Adjust the `dataset, table` strings / `STAGE` in `build_merged_data.py` if your catalog differs. The
   `age`/`sex` columns are matched by `pick_col()` (token-boundary) so minor naming differences are handled.
4. **Check the case/control rules.** `build_merged_data.py::POSITIVE` binarizes `<disease>__curated_phenotype`.
   Verify these category rules against the **real** curated-phenotype categories (the script prints the
   categories it sees) before trusting the labels.
5. **Build the merged table:**
   ```bash
   python build_merged_data.py --disease diabetes
   ```
   This writes `MESHAgents/data/merged_data.csv` (phenotypes X + confounders C + binary label) and
   `MESHAgents/data/structures.json` (region → phenotype groups). Everything downstream reads these two
   files and is identical for synthetic vs real data.

> Note: keep PHI inside the TRE. `MESHAgents/data/` and `results/` are gitignored — do **not** commit real data.

---

## 4. Run the pipeline (3-stage MESHAgents PheWAS)

```bash
cd MESHAgents

# (a) statistics only — no LLM, no key, fast (sanity check)
MESH_DRY_RUN=1 OFFLINE_LLM=1 python src/main_bodycomp.py

# (b) full protocol with the configured provider (OpenAI or Ollama)
python src/main_bodycomp.py
# Ollama explicitly:
MESH_PROVIDER=ollama MESH_LLM_MODEL=qwen2.5:7b python src/main_bodycomp.py
```

What it does: Stage I per-region valuation → Stage II associative-factor discovery + confounder
identification → confounder-adjusted PheWAS → Stage III sequential multi-agent consensus (≤10 rounds) →
`f_AP` → Auto-PheWAS metrics. Writes to `MESHAgents/results/`:
`discovered_phenotypes_*.json`, `consensus_transcript_*.json`, `phenotype_association_*.csv`,
`<region>_analysis_*.json`, `gpt_analysis_*.json`, `analysis_results_*.json`.

---

## 5. Evaluation, figures, baselines, ablations (from repo root)

```bash
cd ..
python evaluate_diagnosis.py     # Track-2: expert vs discovered (+conf), stratified 5-fold CV, AUC+Recall
python phewas_figures.py         # Fig 1(a) overlap + Fig 1(b) heatmap + Fig 3 ROC -> results/*.png
```
```bash
cd MESHAgents                    # baselines/ablations run from here so .env loads
python ../phewas_baselines.py    # Table-1 "Independent LLMs" (provider-aware; ollama uses MESH_LLM_MODEL)
python ../phewas_ablation.py     # Table-1 ablations (full / w-o Stage III / w-o Mem.&Tool) — full LLM run, slow
```

---

## 6. End-to-end quick reference

```bash
# one-time
git clone https://github.com/aj-das-research/pheno26.git && cd pheno26
./setup.sh && source .venv/bin/activate
# edit .env: set MESH_PROVIDER + key/model

# synthetic
python make_synthetic.py
python build_merged_data.py --disease diabetes

# real Pheno data instead: edit pheno_io.py USE_SYNTHETIC=False, configure pheno_utils, then
# python build_merged_data.py --disease diabetes

# run
cd MESHAgents && python src/main_bodycomp.py && cd ..
python evaluate_diagnosis.py && python phewas_figures.py
```

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OPENAI_API_KEY not found ... raise` | OpenAI mode with no key. Add it to `.env`, or use `MESH_PROVIDER=ollama`, or `MESH_DRY_RUN=1`. |
| `Cannot reach Ollama at .../v1` | `ollama serve` not running, or wrong `OLLAMA_HOST`. |
| `Ollama is up but missing model(s)` | `ollama pull qwen2.5:7b nomic-embed-text`. |
| `Missing required columns: {LVEF (%) ...}` | You ran `src/main.py` (upstream cardiac). Use `src/main_bodycomp.py`. |
| `Unable to find a usable engine ... parquet` | `pip install pyarrow` (already in `requirements.txt`; re-run `./setup.sh`). |
| `merged_data.csv` not found | Run `build_merged_data.py` first. |
| Real data: `KeyError`/empty on a dataset/table | dataset/table name mismatch — adjust the `get_df(...)` strings / `STAGE` in `build_merged_data.py` to your catalog. |
| 401 on region narratives under `MESH_DRY_RUN=1` | expected if `OFFLINE_LLM` unset + dummy key; add `OFFLINE_LLM=1` for a fully offline run. |

---

## 8. On the Pheno TRE VM — the two flags

```python
# pheno_io.py
USE_SYNTHETIC = False        # real data via pheno_utils.PhenoLoader
```
```bash
# shell / .env
MESH_PROVIDER=ollama         # local models, no internet, no OpenAI key
ollama serve & ; ollama pull qwen2.5:7b nomic-embed-text
python MESHAgents/src/main_bodycomp.py
```
No upstream files (`agents.py`, `main.py`, `config.py`) need editing for either switch.
