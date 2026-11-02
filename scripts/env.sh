# source me from the repo root: shared env for heavy steps. Reuses the sibling libero-vla-repro venv read-only.
export HERE="$PWD"
export REPRO="$HERE/../libero-vla-repro"
export PY="$REPRO/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$HERE/scripts:$PYTHONPATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$HERE/.libero_cfg" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
heavy() { nice -n 19 ionice -c3 "$@"; }
