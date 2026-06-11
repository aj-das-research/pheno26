"""Single source of truth for the LLM provider (OpenAI <-> local Ollama), switched by ONE env var.

    MESH_PROVIDER=openai   (default)  -> api.openai.com via your OPENAI_API_KEY
    MESH_PROVIDER=ollama              -> local Ollama (OpenAI-compatible) at OLLAMA_HOST + /v1

Ollama speaks the OpenAI API, so the existing `openai` SDK works against it by pointing `base_url` at the
local server with a dummy key. Every client-creation site in the codebase routes through this module, so
flipping `MESH_PROVIDER` flips the entire pipeline (region narratives, Stage I/III LLM calls, chief
synthesis, embeddings, and phewas_baselines). Model selection: MESH_LLM_MODEL / MESH_EMBED_MODEL.

Designed for the HPP/Pheno TRE VM (no internet, Ollama available). Upstream files (config.py, agents.py,
main.py) are left UNCHANGED: this module's import-time shim sets a dummy OPENAI_API_KEY in ollama mode so
`config.py`'s "key or raise" check passes — hence entry points must `import llm_provider` BEFORE `config`.
"""
import os
from dotenv import load_dotenv

OLLAMA_DEFAULT_CHAT = "qwen2.5:7b"          # strong JSON adherence for the agents' structured opinions
OLLAMA_DEFAULT_EMBED = "nomic-embed-text"
OPENAI_DEFAULT_EMBED = "text-embedding-3-small"
_OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def provider() -> str:
    return os.getenv("MESH_PROVIDER", "openai").strip().lower()


def is_ollama() -> bool:
    return provider() == "ollama"


# ---- import-time shim: make config.py importable on a keyless Ollama VM, WITHOUT editing config.py ----
# Only in ollama mode: load .env (mirrors config.py) then setdefault a dummy key. setdefault never
# overrides a real key; gating on ollama means openai mode is never touched (config loads the real key).
if is_ollama():
    load_dotenv()
    os.environ.setdefault("OPENAI_API_KEY", "ollama")


def _ollama_base() -> str:
    return os.getenv("OLLAMA_HOST", _OLLAMA_DEFAULT_HOST).rstrip("/") + "/v1"


def chat_base_url():
    """Explicit override > ollama-derived > None (openai -> SDK default api.openai.com)."""
    explicit = os.getenv("MESH_LLM_BASE_URL")
    if explicit:
        return explicit
    return _ollama_base() if is_ollama() else None


def embed_base_url():
    explicit = os.getenv("MESH_EMBED_BASE_URL") or os.getenv("MESH_LLM_BASE_URL")
    if explicit:
        return explicit
    return _ollama_base() if is_ollama() else None


def chat_model(default=None) -> str:
    """MESH_LLM_MODEL > (ollama? qwen2.5:7b) > caller default (e.g. config.GPT_MODEL)."""
    explicit = os.getenv("MESH_LLM_MODEL")
    if explicit:
        return explicit
    if is_ollama():
        return OLLAMA_DEFAULT_CHAT
    return default if default is not None else "gpt-5-mini-2025-08-07"


def embed_model() -> str:
    explicit = os.getenv("MESH_EMBED_MODEL")
    if explicit:
        return explicit
    return OLLAMA_DEFAULT_EMBED if is_ollama() else OPENAI_DEFAULT_EMBED


def _api_key() -> str:
    # In ollama mode always use a dummy — never send a real OpenAI key to a local endpoint.
    if is_ollama():
        return "ollama"
    return os.getenv("OPENAI_API_KEY", "dummy")


def chat_client():
    """An openai.OpenAI client bound to the active provider's base_url."""
    from openai import OpenAI
    base = chat_base_url()
    return OpenAI(api_key=_api_key(), base_url=base) if base else OpenAI(api_key=_api_key())


def embed_client():
    from openai import OpenAI
    base = embed_base_url()
    return OpenAI(api_key=_api_key(), base_url=base) if base else OpenAI(api_key=_api_key())


def check() -> None:
    """Health check: verify the Ollama endpoint is reachable and the chosen models are pulled.
    No-op in openai mode. Raises a clear, actionable RuntimeError otherwise (uses the openai SDK only)."""
    if not is_ollama():
        return
    want_chat, want_embed = chat_model(), embed_model()
    try:
        available = {m.id for m in chat_client().models.list().data}
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {chat_base_url()}: {e}\n"
            f"Start it with `ollama serve` and ensure OLLAMA_HOST is correct."
        ) from e
    # Ollama reports tags like 'qwen2.5:7b'; tolerate ':latest' shorthand.
    def _present(name):
        return name in available or f"{name}:latest" in available or any(
            a == name or a.split(":")[0] == name.split(":")[0] for a in available)
    missing = [m for m in (want_chat, want_embed) if not _present(m)]
    if missing:
        raise RuntimeError(
            f"Ollama is up but missing model(s): {missing}. Pull them with:\n"
            f"  ollama pull {' '.join(dict.fromkeys([want_chat, want_embed]))}"
        )
