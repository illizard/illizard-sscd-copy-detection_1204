# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.




# head 1개씩 떨군거 다시 불러와서 
# 오름 차순 정렬 하고 
# 다시 돌리는 코드 



import logging
import os
import torch
import pytorch_lightning as pl
from classy_vision.generic.distributed_util import get_rank, get_world_size, barrier
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.plugins import DDPSpawnPlugin
from torch.utils.data import DataLoader
from sscd.train_og import SSCD
from sscd.models.model_og import Model
from sscd.lib.util import call_using_args, parse_bool

from utils.make_file import make_dict_score_based, make_dict_similarity_based, make_dict_random_based
# from utils.pruned_flpos_count import perform_head_pruning_and_cal_flops
from torch.quantization import quantize_dynamic, quantize

from neural_compressor.experimental import Quantization, common
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor.quantization import fit


logger = logging.getLogger("inference.py")
logger.setLevel(logging.INFO)


class InferenceModel(pl.LightningModule):
    """Wraps a model for inference."""

    def __init__(self, model, metadata_keys):
        super().__init__()
        self.model = model
        self.metadata_keys = metadata_keys
        # #modi
        # self.block_idx = args.block_idx
        # self.head_idx = args.head_idx
        
    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        input = batch["input"]
        batch = {k: v for (k, v) in batch.items() if k in self.metadata_keys}
        batch["embeddings"] = self(input)

        # Workaround for a CUDA synchronization bug in PyTorch Lightning.
        # Fixed upstream:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11287
        batch = {k: v.cpu() for (k, v) in batch.items()}

        return batch


class Inference:
    @classmethod
    def add_parser_args(cls, parser):
        parser.add_argument("--checkpoint")
        parser.add_argument("--features")
        parser.add_argument("--model_state")
        parser.add_argument("--output_path", required=True)
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--accelerator", default="auto")
        parser.add_argument("--nodes", default=1, type=int)
        # parser.add_argument("--workers", default=10, type=int)
        parser.add_argument("--workers", default=0, type=int)
        parser.add_argument(
            "--size", default=288, type=int, help="Image size for inference"
        )
        parser.add_argument("--preserve_aspect_ratio", default=False, type=parse_bool)
        ################
        # MODI 
        ################
        parser.add_argument("--head_drop_ratio", default=0.1, type=float)
        parser.add_argument("--csv_in_path", required=True, type=str)

        # These options are only used if --model_state is provided.
        Model.add_arguments(parser)
    
    @classmethod
    def inference(cls, args, dataset, base_name="predictions"):
        if args.features:
            logger.info("Loading features")
            if os.path.exists(args.features):
                features_fn = args.features
            else:
                features_fn = f"{args.features}/{base_name}.pt"
            outputs = torch.load(features_fn, map_location=torch.device("cpu"))
        elif args.checkpoint or args.model_state:
            logger.info("Loading model")
            if args.checkpoint:
                pl_model = SSCD.load_from_checkpoint(
                    args.checkpoint, map_location=torch.device("cpu")
                )
                cls.perform_head_pruning(pl_model.model, args, 64)  # 헤드 프루닝 적용
                
            else:
                model = call_using_args(Model, args)
                
                # 헷갈리지 말 것! 여기서 프리트레인 모델 가져옴, model_og에서 가져오는 거 아님
                state_tmp = torch.load(args.model_state, map_location=torch.device("cpu"))
                
                new_state_dict = {key.replace('model.', ''): value for key, value in state_tmp['state_dict'].items()}
                model.load_state_dict(new_state_dict)
                # MODI 양자화 설정
                model = cls.perform_head_pruning(model, args, 64)  # 헤드 프루닝 적용
                pl_model = InferenceModel(model, ["image_num", "split", "instance_id"])

            logger.info("Creating dataloader")
            dataloader = DataLoader(
                dataset,
                batch_size=1 if args.preserve_aspect_ratio else 256,
                # num_workers=args.workers,
                num_workers=0,
                persistent_workers=(
                    args.workers > 0
                ),  # unnecessary here, but silences warning
            )
            writer = InferenceWriter(args.output_path, base_name)
            #### OG DDP ####
            # trainer = pl.Trainer(
            #     devices=args.gpus,
            #     num_nodes=args.nodes,
            #     accelerator=args.accelerator,
            #     default_root_dir=args.output_path,
            #     strategy=DDPSpawnPlugin(find_unused_parameters=False),
            #     # strategy='ddp',
            #     callbacks=[writer],
            #     log_every_n_steps=1,
            # )
            
            #### MODI for single cpu ####
            trainer = pl.Trainer(
                devices=1,
                num_nodes=1,
                accelerator='cpu',
                default_root_dir=args.output_path,
                strategy=None,
                callbacks=[writer],
                log_every_n_steps=1,
            )
            logger.info("Starting inference")
   
            # print(pl_model)
            print(pl_model.to("cpu"))
            print(f"trainer.devices is {trainer.devices}")
            print(f"trainer.num_nodes is {trainer.num_nodes}")
            print(f"trainer.accelerator is {trainer.accelerator}")
            trainer.predict(pl_model, dataloaders=dataloader)
            logger.info("Loading features")
            outputs = writer.read()
        else:
            raise ValueError("Either --checkpoint or --features is required")

        logger.info("Deduplication")
        outputs = SSCD.dedup_outputs(outputs)

        return outputs
    
    ##################################################
    # MODI
    # 
    # 이 메소드는 각 블록과 헤드 인덱스 쌍에 대해 반복적으로 가중치와 바이어스를 수정합니다. blocks_and_heads 파라미터는 튜플의 리스트로, 각 튜플은 (블록 인덱스, 헤드 인덱스) 형식입니다.
    ##################################################

    # 이상없음, 값 전부 일일히 확인해봄 
    @staticmethod
    def perform_head_pruning(model, args, head_size):
        model.eval()
        if args.based == 'score':
            sorted_data_dict = make_dict_score_based(args.head_drop_ratio, args.csv_in_path)
            
            for block_index, head_index in sorted_data_dict:
                # qkv 가중치와 바이어스에 접근
                qkv_weight = model.backbone.blocks[block_index].attn.qkv.weight.data
                qkv_bias = model.backbone.blocks[block_index].attn.qkv.bias.data

                # 가중치와 바이어스를 Query, Key, Value로 분할
                dim = qkv_weight.shape[0] // 3
                q_weight, k_weight, v_weight = qkv_weight[:dim], qkv_weight[dim:2*dim], qkv_weight[2*dim:]
                q_bias, k_bias, v_bias = qkv_bias[:dim], qkv_bias[dim:2*dim], qkv_bias[2*dim:]

                # 특정 헤드의 가중치와 바이어스를 0으로 설정
                start_idx = head_index * head_size
                end_idx = (head_index + 1) * head_size
                q_weight[start_idx:end_idx] = 0
                k_weight[start_idx:end_idx] = 0
                v_weight[start_idx:end_idx] = 0
                q_bias[start_idx:end_idx] = 0
                k_bias[start_idx:end_idx] = 0
                v_bias[start_idx:end_idx] = 0

                # 수정된 가중치와 바이어스를 원래 qkv 가중치와 바이어스에 다시 할당
                model.backbone.blocks[block_index].attn.qkv.weight.data = torch.cat([q_weight, k_weight, v_weight], 0)
                model.backbone.blocks[block_index].attn.qkv.bias.data = torch.cat([q_bias, k_bias, v_bias], 0)

            # 모델 양자화
            model = quantize_dynamic(
                model,  # 양자화할 모델  # 양자화할 레이어 유형
                qconfig_spec={torch.nn.Linear, torch.nn.Conv2d},  # 사용할 데이터 타입
                dtype=torch.qint8  # 사용할 데이터 타입
            )
        # return model  
        return model.to("cpu")
                                     
def coalesce_outputs(outputs):
    keys = outputs[0].keys()
    return {k: torch.cat([out[k] for out in outputs]) for k in keys}


class InferenceWriter(BasePredictionWriter):
    def __init__(self, output_path: str, filename: str):
        super().__init__("epoch")
        self.output_path = output_path
        self.filename = filename
        self.output_file = os.path.join(self.output_path, f"{filename}.pt")

    def _rank_fn(self, i):
        return os.path.join(self.output_path, f"{self.filename}_rank_{i}.pt")

    def write_on_epoch_end(self, trainer, module, predictions, batch_indices):
        rank = get_rank()
        assert len(predictions) == 1
        predictions = predictions[0]
        outputs = coalesce_outputs(predictions)
        logger.info(
            "Writing %d outputs for worker %d", outputs["embeddings"].size(0), rank
        )
        torch.save(outputs, self._rank_fn(rank))
        del outputs
        logger.info("Rank %d done. Waiting for peers.", rank)
        barrier()
        if rank == 0:
            logger.info("Combining prediction outputs.")
            worker_output_fns = [self._rank_fn(i) for i in range(get_world_size())]
            worker_outputs = [torch.load(fn) for fn in worker_output_fns]
            outputs = coalesce_outputs(worker_outputs)
            del worker_outputs
            torch.save(outputs, self.output_file)
            logger.info("Save completed.")
            for fn in worker_output_fns:
                os.remove(fn)

    def read(self):
        return torch.load(self.output_file)