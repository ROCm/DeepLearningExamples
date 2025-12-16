# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from typing import List

import torch
import torch.nn as nn
import dgl
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.arguments import PARSER
from se3_transformer.runtime.callbacks import BaseCallback
from se3_transformer.runtime.loggers import DLLogger, WandbLogger, LoggerCollection
from se3_transformer.runtime.utils import to_cuda, get_local_rank


@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             callbacks: List[BaseCallback],
             args):
    print("I am inside evaluate")
    model.eval()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch', desc=f'Evaluation',
                         leave=False, disable=(args.silent or get_local_rank() != 0)):
        *input, target = to_cuda(batch)

        for callback in callbacks:
            callback.on_batch_start()

        if (args.amp):
            with torch.amp.autocast('cuda'):
                pred = model(*input)
                  # ---- Demo print for the first batch only ----
                if i == 0 and get_local_rank() == 0:
                    print("\n===== DEMO: Single Batch =====")
                    for idx, x in enumerate(input):
                        print(f"\nINPUT[{idx}] type: {type(x)}")
                        if isinstance(x, torch.Tensor):
                            print("  Tensor shape:", x.shape)
                            print("  Tensor dtype:", x.dtype)
                            #print("  First 3 elements:", x[:3] if x.numel() >= 3 else x)
                            print("  First 3 elements:", x)
                        elif isinstance(x, dgl.DGLGraph):
                            print("  DGLGraph properties:")
                            print("    num nodes:", x.num_nodes())
                            print("    num edges:", x.num_edges())
                            # print all available node and edge features
                            if x.ndata:
                                print("    node features keys:", list(x.ndata.keys()))
                                for key, val in x.ndata.items():
                                    print(f"      {key}: shape {val.shape}, dtype {val.dtype}")
                            if x.edata:
                                print("    edge features keys:", list(x.edata.keys()))
                                for key, val in x.edata.items():
                                    print(f"      {key}: shape {val.shape}, dtype {val.dtype}")
                        elif isinstance(x, dict):
                            print("  Dict input keys:", list(x.keys()))
                            for key, val in x.items():
                                if isinstance(val, torch.Tensor):
                                    print(f"    {key}: shape {val.shape}, dtype {val.dtype}")
                                else:
                                    print(f"    {key}: {val}")
                        else:
                            print("  Unknown input type:", type(x))

                    # Print target and prediction
                    if isinstance(target, torch.Tensor):
                        #print("\nTARGET (first 3 elements):", target[:min(3, len(target))].detach().cpu())
                        print("\nTARGET (first 3 elements):", target.detach().cpu())
                    else:
                        print("\nTARGET:", target)

                    if isinstance(pred, torch.Tensor):
                        #print("PRED (first 3 elements):", pred[:min(3, len(pred))].detach().cpu())
                        print("PRED (first 3 elements):", pred.detach().cpu())
                    else:
                        print("PRED:", pred)

                    print("==============================\n")
                for callback in callbacks:
                    callback.on_validation_step(input, target, pred)
        else:
            pred = model(*input)
            for callback in callbacks:
                callback.on_validation_step(input, target, pred)


if __name__ == '__main__':
    from se3_transformer.runtime.callbacks import QM9MetricCallback, PerformanceCallback
    from se3_transformer.runtime.utils import init_distributed, seed_everything
    from se3_transformer.model import SE3TransformerPooled, Fiber
    from se3_transformer.data_loading import QM9DataModule
    import torch.distributed as dist
    import logging
    import sys

    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()
    print(args)

    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)

    logging.info('====== SE(3)-Transformer ======')
    logging.info('|  Inference on the test set  |')
    logging.info('===============================')

    if not args.benchmark and args.load_ckpt_path is None:
        logging.error('No load_ckpt_path provided, you need to provide a saved model to evaluate')
        sys.exit(1)

    if args.benchmark:
        logging.info('Running benchmark mode with one warmup pass')

    if args.seed is not None:
        seed_everything(args.seed)

    major_cc, minor_cc = torch.cuda.get_device_capability()

    if args.amp:
        amp_str = 'amp_'
    else:
        amp_str = ''

    loggers = [DLLogger(save_dir=args.log_dir, filename='single_gpu_inference_'+amp_str+args.dllogger_name)]
    if args.wandb:
        loggers.append(WandbLogger(name=f'QM9({args.task})', save_dir=args.log_dir, project='se3-transformer'))
    logger = LoggerCollection(loggers)
    datamodule = QM9DataModule(**vars(args))
    # ADD THIS BLOCK
    loader = datamodule.test_dataloader()
    dataset = loader.dataset
    graph, target = dataset[0]

    print("\n===== DATASET[0] GRAPH DETAILS =====")

    print("\n--- BASIC INFO ---")
    print("Nodes:", graph.num_nodes())
    print("Edges:", graph.num_edges())

    print("\n--- NODE DATA (ndata) ---")
    for key, value in graph.ndata.items():
        print(f"Key: {key}")
        print("  shape:", value.shape)
        print("  dtype:", value.dtype)
        # print first few rows only
        #print("  first 3 entries:\n", value[:3])
        print("  first 3 entries:\n", value)

    print("\n--- EDGE DATA (edata) ---")
    for key, value in graph.edata.items():
        print(f"Key: {key}")
        print("  shape:", value.shape)
        print("  dtype:", value.dtype)
        # print first few rows only
        #print("  first 3 entries:\n", value[:3])
        print("  first 3 entries:\n", value)

    print("\n--- TARGET ---")
    print("Shape:", target.shape)
    print("Value:", target)

    print("===============================\n")
    model = SE3TransformerPooled(
        fiber_in=Fiber({0: datamodule.NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: datamodule.EDGE_FEATURE_DIM}),
        output_dim=1,
        tensor_cores=(args.amp and major_cc >= 7) or major_cc >= 8,  # use Tensor Cores more effectively
        **vars(args)
    )
    callbacks = [QM9MetricCallback(logger, targets_std=datamodule.targets_std, prefix='test')]

    model.to(device=torch.cuda.current_device())
    if args.load_ckpt_path is not None:
        checkpoint = torch.load(str(args.load_ckpt_path), map_location={'cuda:0': f'cuda:{local_rank}'}, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])

    if is_distributed:
        nproc_per_node = torch.cuda.device_count()
        affinity = gpu_affinity.set_affinity(local_rank, nproc_per_node, scope='socket')
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()

    torch.set_float32_matmul_precision('high')

    test_dataloader = datamodule.test_dataloader() if not args.benchmark else datamodule.train_dataloader()
    if not args.benchmark:
        print("Line before evaluate&&&&")
        evaluate(model,
                 test_dataloader,
                 callbacks,
                 args)

        for callback in callbacks:
            callback.on_validation_end()

    else:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        callbacks = [PerformanceCallback(
            logger, args.batch_size * world_size,
            warmup_epochs=1 if args.epochs > 1 else 0,
            mode='inference'
        )]
        for _ in range(args.epochs):
            evaluate(model,
                     test_dataloader,
                     callbacks,
                     args)
            callbacks[0].on_epoch_end()

        callbacks[0].on_fit_end()
