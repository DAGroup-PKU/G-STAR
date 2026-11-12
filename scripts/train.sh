#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python3}
export USE_TF=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29570}
exec "$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --module gstar.train "$@"
