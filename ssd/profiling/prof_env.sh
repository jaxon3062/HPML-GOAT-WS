# Profiling environment (no API key here: W&B uses your own login in ~/.netrc)
export PATH="$HOME/.local/bin:$PATH"
export SSD_CUDA_ARCH=8.0
export SSD_HF_CACHE=/data/hf/hub
export SSD_DATASET_DIR=/data/hf_datasets/processed_datasets
export WANDB_ENTITY=HPML-GOAT
export WANDB_PROJECT=ssd
export WANDB_DIR=$HOME/ssd-prof/wandb_runs
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor_cache
mkdir -p "$WANDB_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
# /etc/nccl.conf has a 4-GPU Azure graph; 2/3-GPU runs need the empty override (same as teammate)
export NCCL_GRAPH_FILE=/data/nccl_empty_graph.xml
source $HOME/ssd-prof/.venv/bin/activate
