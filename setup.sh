#!/usr/bin/env bash
# One-command environment setup for pheno26.
# Creates a Python 3.10 virtualenv at ./.venv (project root) and installs all deps.
# Uses `uv` if available (fast); falls back to the stdlib `venv` + pip.
#
#   ./setup.sh           # create .venv and install requirements
#   source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"

PYVER=3.10

if command -v uv >/dev/null 2>&1; then
  echo "==> creating .venv with uv (Python ${PYVER})"
  uv venv --python "${PYVER}" .venv
  echo "==> installing requirements"
  uv pip install --python .venv/bin/python -r requirements.txt
else
  echo "==> uv not found; using python${PYVER} venv + pip"
  PY="$(command -v python${PYVER} || command -v python3)"
  "${PY}" -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi

# seed .env from the template if missing (never overwrites an existing .env)
if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> created .env from .env.example — edit it and add your OPENAI_API_KEY"
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then:                 python make_synthetic.py && python build_merged_data.py --disease diabetes"
