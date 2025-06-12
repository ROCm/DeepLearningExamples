#!/bin/bash

bash scripts/benchmark_train.sh 120
bash scripts/benchmark_train.sh 240
bash scripts/benchmark_train.sh 120 --amp
bash scripts/benchmark_train.sh 240 --amp
bash scripts/benchmark_train_multi_gpu.sh 120
bash scripts/benchmark_train_multi_gpu.sh 240
bash scripts/benchmark_train_multi_gpu.sh 120 --amp
bash scripts/benchmark_train_multi_gpu.sh 240 --amp
bash scripts/benchmark_inference.sh 400
bash scripts/benchmark_inference.sh 400 --amp 