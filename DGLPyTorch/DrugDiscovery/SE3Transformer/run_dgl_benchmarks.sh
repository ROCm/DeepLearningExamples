#!/bin/bash

set -euo pipefail

# Log file
LOGFILE="dgl_se3_benchmarks_$(hostname).log"
echo "Running benchmarks on $(hostname)" > $LOGFILE

# Print GPU-specific information
if [ $(which amd-smi) ]; then
    amd-smi static | grep "MARKET_NAME" | tee -a $LOGFILE
elif [ $(which nvidia-smi) ]; then
    nvidia-smi | tee -a $LOGFILE
fi

echo "DGL version: $(python -c "import dgl; print(dgl.__version__)")" | tee -a $LOGFILE

# Run benchmarks
export DGLBACKEND=pytorch
bash scripts/benchmark_train.sh 120 2>&1 | tee -a $LOGFILE
bash scripts/benchmark_train.sh 240 2>&1 | tee -a $LOGFILE
bash scripts/benchmark_train_multi_gpu.sh 120 2>&1 | tee -a $LOGFILE
bash scripts/benchmark_train_multi_gpu.sh 240 2>&1 | tee -a $LOGFILE
bash scripts/benchmark_inference.sh 400 2>&1 | tee -a $LOGFILE

echo "Benchmarks completed. Results saved to $LOGFILE"  
