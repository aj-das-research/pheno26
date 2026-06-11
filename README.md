# pheno26 — Whole-Body Multi-Agent Phenotype Analysis (MESHAgents on HPP)

Adapting the **MESHAgents** framework (*Multi-Agent Reasoning for Cardiovascular Imaging Phenotype
Analysis*, MICCAI 2025) from single-organ cardiac imaging to **whole-body phenotype analysis** on the
**Human Phenotype Project (HPP)**, developed and validated on synthetic data first.

> **Honest scope.** The upstream repo ([LumaLabAI/MESHAgents](https://github.com/LumaLabAI/MESHAgents),
> MIT) ships the base agents + an unsupervised salience score, but **not** the paper's full 3-stage
> protocol. The faithful 3-stage protocol here (`MESHAgents/src/mesh_core.py` — dynamic memory, Stage I/II/III
> sequential LLM consensus, `f_AP`, Auto-PheWAS metrics) is **our re-implementation from the paper's text**,
> adapted to body composition and validated on **synthetic** data. Absolute numbers are illustrative, not
> a literal reproduction of the paper's UK Biobank cardiac results. See `PROJECT_DOCUMENTATION.md` for the
> full design, faithful/approximated component map, and honest caveats.

## Quickstart

```bash
cd MESHAgents
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install --python .venv/bin/python pandas numpy scipy statsmodels scikit-learn \
  seaborn matplotlib python-dotenv pyarrow "openai>=1.0.0"
cp .env.example .env          # then add your OpenAI key (never commit .env)

cd ..
python make_synthetic.py                         # synthetic HPP-schema parquet tables
python build_merged_data.py --disease diabetes   # -> MESHAgents/data/merged_data.csv

cd MESHAgents
MESH_DRY_RUN=1 OPENAI_API_KEY=dummy python src/main_bodycomp.py   # full protocol, no API calls
# or: python src/main_bodycomp.py                                  # real LLM consensus (gpt-5-mini)

cd ..
python evaluate_diagnosis.py    # Track-2: expert vs discovered (+conf), 5-fold CV, AUC+Recall
python phewas_figures.py        # Fig 1(b) heatmap + Fig 3 ROC
python phewas_baselines.py      # Table-1 independent-LLM baselines  (run from MESHAgents/)
python phewas_ablation.py       # Table-1 ablations (full / w-o Stage III / w-o Mem.&Tool)
```

## What's reproduced vs not

| Paper component | Status |
|---|---|
| 3-stage protocol (memory, Stage I/II/III consensus, `f_AP`) | ✅ re-implemented from the paper |
| Confounder discovery (Stage II) | ✅ |
| Auto-PheWAS Dependency Q / Coverage C | ✅ exact formulas |
| Diagnosis: 5-fold CV, AdaBoost/LDA/SVM, AUC+Recall, expert vs discovered | ✅ |
| Independent-LLM baselines (Table 1) | ✅ |
| Ablations (Table 1) | ✅ (Stage-III ablation is null on synthetic data — documented) |
| Fig 1(a/b), Fig 3 | ✅ |
| MedAgents / RareAgents baselines | ❌ external codebases (out of scope) |
| UK Biobank cardiac data | ❌ we use synthetic body-composition data |

## Attribution & license

`MESHAgents/` is adapted from [LumaLabAI/MESHAgents](https://github.com/LumaLabAI/MESHAgents)
(© 2025 Mengyun, MIT License — see `MESHAgents/LICENSE`). Paper: Zhang, Qiao et al., *Multi-Agent
Reasoning for Cardiovascular Imaging Phenotype Analysis*, MICCAI 2025 ([arXiv:2507.03460](https://arxiv.org/abs/2507.03460)).
