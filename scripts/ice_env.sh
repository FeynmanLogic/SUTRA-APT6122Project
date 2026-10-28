#!/usr/bin/env bash
# Run from notebook or terminal; load modules in the same shell as the command.
set -eo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo 'Run this wrapper on Linux inside your PACE-ICE compute allocation.' >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'No SLURM_JOB_ID: launch an ICE compute-node Jupyter/VS Code session first.' >&2
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == ece4122-6122 || "${CONDA_PREFIX:-}" == */ece4122-6122 ]]; then
    echo 'The course CUDA setup excludes the ece4122-6122 conda build environment. Start a fresh system-toolchain session.' >&2
    exit 2
fi
if [[ "${SUTRA_WITH_CUDA:-0}" == 1 ]]; then
    if ! type module >/dev/null 2>&1; then
        for init_file in /etc/profile.d/modules.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
            if [[ -f "$init_file" ]]; then
                source "$init_file"
                if type module >/dev/null 2>&1; then break; fi
            fi
        done
    fi
    if ! type module >/dev/null 2>&1; then
        echo 'Cannot initialize the module command; use the course compute-session terminal.' >&2
        exit 2
    fi
    module load "${SUTRA_CUDA_MODULE:-cuda}"
    command -v nvcc >/dev/null
fi
export CC="${SUTRA_CC:-/usr/bin/gcc}"
export CXX="${SUTRA_CXX:-/usr/bin/g++}"
if [[ ! -x "$CC" || ! -x "$CXX" ]]; then
    echo 'Compiler path missing; set SUTRA_CC/SUTRA_CXX to absolute supported system/module compiler paths.' >&2
    exit 2
fi
export CUDAHOSTCXX="$CXX"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/ice_env.sh command [arguments...]' >&2
    exit 2
fi
exec "$@"
