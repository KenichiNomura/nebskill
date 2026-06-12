#!/bin/bash
# SLURM job template for a single NEB reaction.
# Variables injected by submit.py via --export:
#   REACTION_ID  — integer reaction index
#   CONFIG       — path to neb_defaults.yaml (relative to NEB_ROOT)
#   NEB_ROOT     — absolute path to the nebskill repo root
#
# Default SLURM directives are set in neb_defaults.yaml (batch section)
# and passed as flags by submit.py; do NOT hardcode #SBATCH lines here.

set -euo pipefail

echo "=== NEB job start: $(date) ==="
echo "  Host:        $(hostname)"
echo "  Reaction ID: ${REACTION_ID}"
echo "  Config:      ${CONFIG}"
echo "  NEB root:    ${NEB_ROOT}"

# ── environment ────────────────────────────────────────────────────────────
# Activate the shared venv. Adjust path if your venv is elsewhere.
VENV="${NEB_ROOT}/../.venv"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    # try sibling of the project root
    VENV="${NEB_ROOT}/../../.venv"
fi
source "${VENV}/bin/activate"
echo "  Python:      $(which python)  ($(python --version))"

# ── run ────────────────────────────────────────────────────────────────────
cd "${NEB_ROOT}"

python agent/llm_agent.py \
    --reaction-id "${REACTION_ID}" \
    --config "${CONFIG}" \
    --defaults

EXIT_CODE=$?
echo "=== NEB job end: $(date)  exit_code=${EXIT_CODE} ==="
exit ${EXIT_CODE}
