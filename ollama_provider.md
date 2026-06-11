# Local LLM provider (Ollama) — one-switch OpenAI ↔ Ollama

The pipeline can run entirely against **local Ollama** models instead of OpenAI, flipped by a **single
environment variable**. This is built for the **HPP / Pheno AI Trusted Research Environment (TRE) VM**,
which has **no outbound internet** (OpenAI unreachable) but supports Ollama. Ollama exposes an
OpenAI-compatible API at `http://localhost:11434/v1`, so the existing `openai` SDK works against it by
pointing `base_url` at the local server with a dummy key.

All client-creation sites route through one module — [`MESHAgents/src/llm_provider.py`](MESHAgents/src/llm_provider.py) —
so the switch flips the **entire** LLM path: region narratives, Stage I clinical-relevance, Stage III
consensus opinions, the chief synthesis, embeddings, and `phewas_baselines.py`.

## The one switch

```bash
# OpenAI (default) — nothing to set
python MESHAgents/src/main_bodycomp.py

# Local Ollama — ONE change
MESH_PROVIDER=ollama python MESHAgents/src/main_bodycomp.py
```

or set `MESH_PROVIDER=ollama` in `.env` (see `.env.example` Ollama profile).

## Model selection

| Variable | Default (ollama) | Meaning |
|---|---|---|
| `MESH_LLM_MODEL` | `qwen2.5:7b` | chat model — any pulled Ollama tag (`llama3.1:8b`, `qwen2.5:14b`, `mistral`, …) |
| `MESH_EMBED_MODEL` | `nomic-embed-text` | embedding model (`mxbai-embed-large`, `all-minilm`, …) |

```bash
MESH_PROVIDER=ollama MESH_LLM_MODEL=llama3.1:8b python MESHAgents/src/main_bodycomp.py
```

`qwen2.5:7b` is the default because it adheres well to the strict-JSON the agents emit
(`{recommended_phenotypes, top_factor, …}`); weaker models still work via the fallback chain below.

## Full env-var contract

| Var | Default | Meaning |
|---|---|---|
| `MESH_PROVIDER` | `openai` | the one switch: `openai` \| `ollama` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base; provider appends `/v1` (set to a remote host if Ollama runs elsewhere) |
| `MESH_LLM_MODEL` | openai→`GPT_MODEL`; ollama→`qwen2.5:7b` | chat model |
| `MESH_EMBED_MODEL` | openai→`text-embedding-3-small`; ollama→`nomic-embed-text` | embedding model |
| `MESH_LLM_BASE_URL` / `MESH_EMBED_BASE_URL` | unset | explicit endpoint override (wins over the provider default) |
| `MESH_DRY_RUN` | `0` | skip all LLM/embeddings calls (statistical stub) — provider-agnostic |
| `OFFLINE_LLM` | `0` | skip the narrative/synthesis GPT calls |

Base-url precedence: explicit `MESH_LLM_BASE_URL` > ollama-derived (`OLLAMA_HOST + /v1`) > none (openai → SDK default).

## TRE / VM deployment

```bash
# on the VM (no internet):
ollama serve &                                   # start the local server
ollama pull qwen2.5:7b nomic-embed-text          # pull chat + embedding models (one-time)

cd MESHAgents
MESH_PROVIDER=ollama python src/main_bodycomp.py  # no OpenAI key needed
```

- **No OpenAI key required.** In ollama mode `llm_provider` sets a dummy `OPENAI_API_KEY` at import (only
  if one isn't already present) so the upstream `config.py` "key or raise" check passes — **without editing
  any upstream file**. Entry points import `llm_provider` *before* `config` to guarantee this.
- **Remote Ollama**: if the server runs on another host/port, set `OLLAMA_HOST=http://<host>:<port>`.
- `main_bodycomp.py` runs `llm_provider.check()` first (skipped under `MESH_DRY_RUN`/`OFFLINE_LLM`): it
  fails fast with `ollama serve` / `ollama pull <model>` guidance if the server is down or a model is missing.
- The data layer switches separately via `pheno_io.USE_SYNTHETIC = False` (synthetic ↔ `PhenoLoader`).

## How it works (no upstream edits)

- `llm_provider.chat_client()` / `embed_client()` build an `openai.OpenAI` bound to the active provider's
  `base_url`. `chat_model()` / `embed_model()` resolve the model id per provider.
- `mesh_core.ModelClient` / `Embedder` and our `RegionAgent` / `BodyCompChiefAgent` obtain their clients
  from `llm_provider`. The base classes in upstream `agents.py` still build an OpenAI client in
  `__init__`, but our subclasses **override `self.client`/`self.model` right after `super().__init__()`**,
  so `agents.py`, `main.py`, `config.py` remain untouched.

## Limitations & notes

- **Weak-JSON models**: local models are less reliable at strict JSON. Covered by a 3-layer fallback —
  `chat_json` tries `response_format={"type":"json_object"}` then a plain call then regex `{…}` extraction
  (`mesh_core.py`), and `_agent_opinion` falls back to the statistics-grounded ranking if a model returns
  no usable phenotypes. The final `f_AP` selection is statistics-weighted, so noisy opinions don't derail it.
- **Embedding dimensions differ** (OpenAI `text-embedding-3-small`=1536-d, Ollama `nomic-embed-text`=768-d).
  Safe: `AgentMemory` cosine only ever compares vectors from the **same** run/provider (`_cosine` returns 0
  on shape mismatch). Don't mix providers within a single run's memory.
- **Context length**: the `phewas_baselines.py` prompt lists all 152 phenotypes; use a model with ample
  context (`qwen2.5:7b`/`llama3.1:8b` have ≥32k–128k). Small/quantized models may truncate.
- **Throughput**: local inference is slower than the API; Stage III is sequential (≤10 rounds × N regions),
  so a full LLM run takes minutes-to-tens-of-minutes depending on hardware.
- In ollama mode `phewas_baselines.py` defaults its model list to the single `MESH_LLM_MODEL`; override with
  `MESH_BASELINE_MODELS=qwen2.5:7b,llama3.1:8b,mistral` to compare several local models.

## Verify without Ollama installed

```bash
# statistical pipeline, zero LLM calls — proves config.py no longer blocks on a keyless VM
MESH_PROVIDER=ollama MESH_DRY_RUN=1 OFFLINE_LLM=1 python MESHAgents/src/main_bodycomp.py
```
