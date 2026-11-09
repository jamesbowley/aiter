# Source before running this experiment's tuner, replays or checks: the stock aiter
# environment (../aiter_tune/env_stock.sh) without its venv, which no longer exists.
# flydsl 0.3.4.1 (aiter's FlyDSL modules need check_smem_capacity, which /opt/venv's
# 0.4.1 lacks) comes first on PYTHONPATH instead.
export PYTHONPATH=/home/jbowley/flydsl-0.3.4.1:/home/jbowley/repos/gfx950-gluon-tutorials/scripts/triton_deepbind_shim:/home/jbowley/repos/triton/python:/home/jbowley/repos/aiter
unset LLVM_PASS_PLUGIN_PATH
unset TRITON_AMDGCNAS_PLUGIN
# See env_stock.sh: keeps the torch profiler working in the tuner's spawned workers.
export ROCPROFILER_REGISTER_FORCE_LOAD=0
export HIP_VISIBLE_DEVICES=${GPU:-3}
export TVT=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail
# One cache for warm.py and the tuner, so the tuner finds every kernel compiled.
export TRITON_CACHE_DIR=$TVT/triton_cache
