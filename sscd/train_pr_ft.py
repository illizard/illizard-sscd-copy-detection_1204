#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import logging
import os
import numpy as np
import pytorch_lightning as pl
import torch
from datetime import datetime

# Set up our python environment (eg. PYTHONPATH).
from lib import initialize  # noqa

from classy_vision.dataset.transforms import build_transforms
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader

from sscd.datasets.disc import DISCEvalDataset
from sscd.datasets.image_folder import ImageFolder
from sscd.lib.distributed_util import cross_gpu_batch
from sscd.transforms.repeated_augmentation import RepeatedAugmentationTransform
from sscd.transforms.mixup import ContrastiveMixup
from sscd.models.model_pr_ft import Model         # check
from sscd.transforms.settings import AugmentationSetting
from sscd.lib.util import call_using_args, parse_bool

import wandb
from pytorch_lightning.loggers import WandbLogger

from utils.make_file import make_dict_score_based, make_dict_similarity_based, make_dict_random_based

DEBUG = False

if DEBUG:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

#230807 추가
def seed_everything(seed: int = 42):
    # random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    
def add_train_args(parser: ArgumentParser, required=True):
    parser = parser.add_argument_group("Train")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument(
        "--batch_size",
        # default=4096,
        default=256,
        type=int,
        help="The global batch size (across all nodes/GPUs, before repeated augmentation)",
    )
    parser.add_argument("--infonce_temperature", default=0.05, type=float)
    parser.add_argument("--entropy_weight", default=30, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--augmentations",
        default="STRONG_BLUR",
        choices=[x.name for x in AugmentationSetting],
    )
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument(
        "--base_learning_rate",
        # default=30,
        default=0.3,
        type=float,
        help="Base learning rate, for a batch size of 256. Linear scaling is applied.",
    )
    parser.add_argument(
        "--absolute_learning_rate",
        default=None,
        type=float,
        help="Absolute learning rate (overrides --base_learning_rate).",
    )
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--sync_bn", default=True, type=parse_bool)
    parser.add_argument("--mixup", default=False, type=parse_bool)
    parser.add_argument(
        "--train_image_size", default=224, type=int, help="Image size for training"
    )
    parser.add_argument(
        "--val_image_size", default=224, type=int, help="Image size for validation"
    )
    parser.add_argument(
        "--workers", default=10, type=int, help="Data loader workers per GPU process."
    )
    parser.add_argument("--num_sanity_val_steps", default=-1, type=int)
    # MODI
    parser.add_argument("--orthogonality", default='no', type=str)
    parser.add_argument("--lamb", default=0.0, type=float)
    parser.add_argument("--token", default='cls', required=False, choices=['cls', 'pat', 'cls+pat', 'cls+pat+lin','other'], type=str)
    parser.add_argument("--blk",  default='last', required=False, choices=['last', 'all', '6blks'], type=str)
    parser.add_argument("--desc_size",  default='192', required=False, type=str)
    parser.add_argument("--model_idx", default=1, type=int)
    parser.add_argument("--head_drop_ratio", default=0.1, type=float)


def add_data_args(parser, required=True):
    parser = parser.add_argument_group("Data")
    parser.add_argument("--output_path", required=required, type=str)
    # parser.add_argument("--train_dataset_path", required=required, type=str)
    parser.add_argument("--train_dataset_path", default= '/hdd/wi/dataset/DISC2021_mini/', required=required, type=str)
    parser.add_argument("--val_dataset_path", required=False, type=str)
    # MODI
    parser.add_argument("--csv_in_path", required=False, type=str)


parser = ArgumentParser()
Model.add_arguments(parser)
add_train_args(parser)
add_data_args(parser)
seed_everything(42)

class DISCData(pl.LightningDataModule):
    """A data module describing datasets used during training."""

    def __init__(
        self,
        *,
        train_dataset_path,
        val_dataset_path,
        train_batch_size,
        augmentations: AugmentationSetting,
        train_image_size=224,
        val_image_size=224,
        val_batch_size=256,
        workers=10,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.workers = workers
        transforms = augmentations.get_transformations(train_image_size)
        transforms = RepeatedAugmentationTransform(transforms, copies=2)
        self.train_dataset = ImageFolder(self.train_dataset_path, transform=transforms)
        if val_dataset_path:
            print(f" self.val_batch_size  {self.val_batch_size}")
            self.val_dataset = self.make_validation_dataset(
                self.val_dataset_path,
                self.val_batch_size,
                size=val_image_size,
            )
        else:
            self.val_dataset = None

    @classmethod
    def make_validation_dataset(
        cls,
        path,
        include_train=False,
        # include_train=True,
        size=224,
        # preserve_aspect_ratio=True,
        # size=288,
        preserve_aspect_ratio=False,
    ):
        transforms = build_transforms(
            [
                {
                    "name": "Resize",
                    # "size": size if preserve_aspect_ratio else [size, size], #OG
                    "size": [size, size] if preserve_aspect_ratio else [size, size], # for ViTs
                },
                {"name": "ToTensor"},
                {
                    "name": "Normalize",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        )

        return DISCEvalDataset(path, transform=transforms, include_train=include_train)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.workers,
            persistent_workers=True,
        )


class SSCD(pl.LightningModule):
    """Training class for SSCD models."""

    def __init__(self, args, train_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = call_using_args(Model, args)
        use_mixup = args.mixup
        self.mixup = (
            ContrastiveMixup.from_config(
                {
                    "name": "contrastive_mixup",
                    "mix_prob": 0.05,
                    "mixup_alpha": 2,
                    "cutmix_alpha": 2,
                    "switch_prob": 0.5,
                    "repeated_augmentations": 2,
                    "target_column": "instance_id",
                }
            )
            if use_mixup
            else None
        )
        self.infonce_temperature = args.infonce_temperature
        self.entropy_weight = args.entropy_weight
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.lr = args.absolute_learning_rate or (
            args.base_learning_rate * args.batch_size / 256
        )
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.train_steps = train_steps
        
        # [modi] path modi 1011
        self.output_path = args.output_path
        self.backbone = args.backbone
        self.orthogonality = args.orthogonality
        self.lamb = args.lamb
        self.entropy_weight = args.entropy_weight
        self.batch_size = args.batch_size
        self.token = args.token
        self.desc_size = args.desc_size
        self.blk = args.blk
        self.model_idx = args.model_idx
        self.head_drop_ratio = args.head_drop_ratio
        self.csv_in_path = args.csv_in_path
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        optimizer = LARS(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            trust_coefficient=0.001,
            eps=1e-8,
        )

        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs * self.train_steps,
                max_epochs=self.epochs * self.train_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Concatenate copies.
        # print(f"batch {batch.keys()}")
        input = torch.cat([batch["input0"], batch["input1"]])
        instance_ids = torch.cat([batch["instance_id"], batch["instance_id"]])
        # print(f"batch {input.shape}")
        # print(f"batch {instance_ids.shape}")

        if self.mixup:
            batch = self.mixup({"input": input, "instance_id": instance_ids})
            input = batch["input"]
            instance_ids = batch["instance_id"]
        embeddings = self(input)

        return self.loss(embeddings, instance_ids)        
################check
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # 옵티마이저 스텝 실행
        optimizer.step(closure=optimizer_closure)

        # sorted_data_dict와 head_size를 정의해야 합니다. 이 값들은 로직에 따라 결정됩니다.
        sorted_data_dict = make_dict_score_based(self.head_drop_ratio, self.csv_in_path)
        head_size = 64   # 헤드당 디멘션 크기

        for block_idx, head_idx in sorted_data_dict:
            # 특정 인코더 블록의 QKV 가중치와 편향 가져오기
            qkv_weight = self.model.backbone.blocks[block_idx].attn.qkv.weight.data
            qkv_bias = self.model.backbone.blocks[block_idx].attn.qkv.bias.data

            # 가중치와 편향을 Q, K, V로 분할
            dim = qkv_weight.shape[0] // 3
            q_weight, k_weight, v_weight = qkv_weight[:dim], qkv_weight[dim:2*dim], qkv_weight[2*dim:]
            q_bias, k_bias, v_bias = qkv_bias[:dim], qkv_bias[dim:2*dim], qkv_bias[2*dim:]
            
            
            # 특정 헤드의 가중치와 편향을 0으로 설정
            start_index = head_idx * head_size
            end_index = start_index + head_size
            for weight, bias in [(q_weight, q_bias), (k_weight, k_bias), (v_weight, v_bias)]:
                # detach()를 호출하여 원본 텐서에서 분리
                detached_weight = weight.detach()
                detached_bias = bias.detach()

                # 분리된 텐서에 대해 변형 수행
                detached_weight[start_index:end_index, :].zero_()
                detached_bias[start_index:end_index].zero_()

                # 원본 텐서에 분리된 텐서의 값을 다시 할당
                weight[start_index:end_index, :] = detached_weight[start_index:end_index, :]
                bias[start_index:end_index] = detached_bias[start_index:end_index]

                # 그래디언트 업데이트 중지
                weight[start_index:end_index, :].requires_grad = False
                bias[start_index:end_index].requires_grad = False

######################
    def cross_gpu_similarity(self, embeddings, instance_labels, mixup: bool):
        """Compute a cross-GPU embedding similarity matrix.

        Embeddings are gathered differentiably, via an autograd function.

        Returns a tuple of similarity, match_matrix, and indentity tensors,
        defined as follows, where N is the batch size (including copies), and
        W is the world size:
          similarity (N, W*N), float: embedding inner-product similarity between
              this GPU's embeddings, and all embeddings in the global
              (cross-GPU) batch.
          match_matrix (N, W*N), bool: cell [i][j] is True when batch[i] and
              global_batch[j] share input content (take any content from the
              same original image), including trivial pairs (comparing a
              copy with itself).
          identity (N, W*N), bool: cell [i][j] is True only for trivial pairs
              (comparing a copy with itself). This identifies the "diagonal"
              in the global (virtual) W*N x W*N similarity matrix. Since each
              GPU has only a slice of this matrix (to avoid N^2 memory use),
              the "diagonal" requires a bit of logic to identify.
        """
        all_embeddings, all_instance_labels = cross_gpu_batch(
            embeddings, instance_labels
        )
        N = embeddings.size(0)
        M = all_embeddings.size(0)
        R = self.global_rank
        similarity = embeddings.matmul(all_embeddings.transpose(0, 1))
        if mixup:
            # In the mixup case, instance_labels are a NxN distribution
            # describing similarity within a per-GPU batch. We infer all inputs
            # from other GPUs are negatives, and use any inputs with nonzero
            # similarity as positives.
            match_matrix = torch.zeros(
                (N, M), dtype=torch.bool, device=embeddings.device
            )
            match_matrix[:, R * N : (R + 1) * N] = (
                instance_labels.matmul(instance_labels.transpose(0, 1)) > 0
            )
        else:
            # In the non-mixup case, instance_labels are instance ID long ints.
            # We broadcast a `==` operation to translate this to a match matrix.
            match_matrix = instance_labels.unsqueeze(
                1
            ) == all_instance_labels.unsqueeze(0)
        identity = torch.zeros_like(match_matrix)
        identity[:, R * N : (R + 1) * N] = torch.eye(N).to(identity)
        return similarity, match_matrix, identity

    def loss(self, embeddings, instance_labels):
        similarity, match_matrix, identity = self.cross_gpu_similarity(
            embeddings, instance_labels, self.mixup
        )
        non_matches = match_matrix == 0
        nontrivial_matches = match_matrix * (~identity)

        # InfoNCE loss
        small_value = torch.tensor(-100.0).to(
            similarity
        )  # any value > max L2 normalized distance
        max_non_match_sim, _ = torch.where(non_matches, similarity, small_value).max(
            dim=1, keepdim=True
        )
        logits = (similarity / self.infonce_temperature).exp()
        partitions = logits + ((non_matches * logits).sum(dim=1) + 1e-6).unsqueeze(1)
        probabilities = logits / partitions
        if self.mixup:
            infonce_loss = (
                (-probabilities.log() * nontrivial_matches).sum(dim=1)
                / nontrivial_matches.sum(dim=1)
            ).mean()
        else:
            infonce_loss = (
                -probabilities.log() * nontrivial_matches
            ).sum() / similarity.size(0)

        components = {"InfoNCE": infonce_loss}
        loss = infonce_loss # 한 배치당 loss
        if self.entropy_weight:
            # Differential entropy regularization loss.
            closest_distance = (2 - (2 * max_non_match_sim)).clamp(min=1e-6).sqrt()
            entropy_loss = -closest_distance.log().mean() * self.entropy_weight
            components["entropy"] = entropy_loss


        loss = infonce_loss + entropy_loss # OG
        
        # Log stats and loss components.
        with torch.no_grad():
            stats = {
                "positive_sim": (similarity * nontrivial_matches).sum()
                / nontrivial_matches.sum(),
                "negative_sim": (similarity * non_matches).sum() / non_matches.sum(),
                "nearest_negative_sim": max_non_match_sim.mean(),
                "center_l2_norm": embeddings.mean(dim=0).pow(2).sum().sqrt(),
            }
        self.log_dict(stats, on_step=False, on_epoch=True)
        self.log_dict(components, on_step=True, on_epoch=True, prog_bar=True)
        if self.logger:
            # OG code 
            # self.logger.experiment.add_scalars(
            #     "loss components",
            #     components,
            #     global_step=self.global_step,
            # )

            # self.logger.experiment.add_scalars(
            #     "similarity stats",
            #     stats,
            #     global_step=self.global_step,
            # )
            
            # 231007
            self.logger.experiment.log({
                "loss components": components,
                "global_step": self.global_step
            })

            self.logger.experiment.log({
                "similarity stats": stats,
                "global_step": self.global_step
            })

        return loss

    def validation_step(self, batch, batch_idx):
        input = batch["input"]
        metadata_keys = ["image_num", "split", "instance_id"]
        batch = {k: v for (k, v) in batch.items() if k in metadata_keys}
        batch["embeddings"] = self(input)

        return batch

    def validation_epoch_end(self, outputs):
        keys = ["embeddings", "image_num", "split", "instance_id"]
        outputs = {k: torch.cat([out[k] for out in outputs]) for k in keys}
        outputs = self._gather(outputs) # 모든 쿼리, 레퍼런스 이미지
        outputs = self.dedup_outputs(outputs)
        if self.current_epoch == 0:
            self.print(
                "Eval dataset size: %d (%d queries, %d index)"
                % (
                    outputs["split"].shape[0],
                    (outputs["split"] == DISCEvalDataset.SPLIT_QUERY).sum(),
                    (outputs["split"] == DISCEvalDataset.SPLIT_REF).sum(),
                )
            )
        dataset: DISCEvalDataset = self.trainer.datamodule.val_dataset
        metrics = dataset.retrieval_eval(
            outputs["embeddings"],
            outputs["image_num"],
            outputs["split"],
        )
        metrics = {k: 0.0 if v is None else v for (k, v) in metrics.items()}
        self.log_dict(metrics, on_epoch=True)


    def on_train_epoch_end(self):
        metrics = []
        for k, v in self.trainer.logged_metrics.items():
            if k.endswith("_step"):
                continue
            if k.endswith("_epoch"):
                k = k[: -len("_epoch")]
            if torch.is_tensor(v):
                v = v.item()
            metrics.append(f"{k}: {v:.3f}")
        metrics = ", ".join(metrics)
        self.print(f"Epoch {self.current_epoch}: {metrics}")

    def on_train_end(self):
        if self.global_rank != 0:
            return
        if not self.logger:
            return
        # OG code
        # path = os.path.join(self.logger.log_dir, "model_torchscript.pt")
        
        # check
        # path = os.path.join(self.output_path, f"cls_score_{self.head_drop_ratio}[model_{self.model_idx}]_{self.backbone}_{self.blk}_{self.orthogonality}_{self.lamb}_{self.token}_{self.desc_size}_ep{self.epochs}_dims_model_torchscript.pt")      
        path = os.path.join(self.output_path, f"cls_orderfirst_{self.head_drop_ratio}[model_{self.model_idx}]_{self.backbone}_{self.blk}_{self.orthogonality}_{self.lamb}_{self.token}_{self.desc_size}_ep{self.epochs}_dims_model_torchscript.pt")      
        # path = os.path.join(self.output_path, f"[og_clspat]_OFFL_VIT_TINY_{self.blk}_{self.orthogonality}_{self.token}_{self.desc_size}dims_model_torchscript.pt")      

        # 나중에 할것
        #path = os.path.join(self.output_path, f"{self.backbone}_{self.blk}_ortho:{self.orthogonal}_{self.lamb}_ent:{self.entropy_weight}_{self.token}_{self.dims}dim_model_torchscript.pt")
        self.save_torchscript(path)
        wandb.save(self.output_path)
    
    def save_torchscript(self, filename):
        self.eval()
        #input = torch.randn((1, 3, 64, 64), device=self.device)
        input = torch.randn((1, 3, 224, 224), device=self.device) #
        # OG 
        script = torch.jit.trace(self.model, input)
        # script = torch.jit.script(self.model)
        torch.jit.save(script, filename)

    def _gather(self, batch):
        batch = self.all_gather(move_data_to_device(batch, self.device))
        return {
            k: v.reshape([-1] + list(v.size()[2:])).cpu() for (k, v) in batch.items()
        }

    @staticmethod
    def dedup_outputs(outputs, key="instance_id"):
        """Deduplicate dataset on instance_id."""
        idx = np.unique(outputs[key].numpy(), return_index=True)[1]
        outputs = {k: v.numpy()[idx] for (k, v) in outputs.items()}
        assert np.unique(outputs[key]).size == outputs["instance_id"].size
        return outputs

    def predict_step(self, batch, batch_idx):
        batch = self.validation_step(batch, batch_idx)

        # Workaround for a CUDA synchronization bug in PyTorch Lightning.
        # Fixed upstream:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11287
        batch = {k: v.cpu() for (k, v) in batch.items()}

        return batch


def main(args):
    # Initialize WandB
    c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')
    # wandb_logger = WandbLogger(name=f'[og_clspat]_OFFL_VIT_TINY_{args.blk}_{args.orthogonality}_{args.token}_{args.dims}_{c_date}_{c_time}', 
	# 		                project='exp_comp')
    # check
    wandb_logger = WandbLogger(name=f'cls_orderfirst_ratio_{args.head_drop_ratio}[model_{args.model_idx}]_{args.backbone}_{args.blk}_{args.orthogonality}_{args.lamb}_{args.token}_{args.desc_size}_ep{args.epochs}_{c_date}_{c_time}', 
			                project='1130_final_prft_orderfirst')
    world_size = args.nodes * args.gpus
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size}) must be a multiple of "
            f"the number of GPUs ({world_size})."
        )
    data = DISCData(
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        train_batch_size=args.batch_size // world_size,
        train_image_size=args.train_image_size,
        val_image_size=args.val_image_size,
        augmentations=AugmentationSetting[args.augmentations],
        workers=args.workers,
    )
    model = SSCD(
        args,
        train_steps=len(data.train_dataset) // args.batch_size,
    )
    # print(model)
    trainer = pl.Trainer(
        devices=args.gpus,
        num_nodes=args.nodes,
        accelerator=args.accelerator,
        max_epochs=args.epochs,
        sync_batchnorm=args.sync_bn,
        default_root_dir=args.output_path,
        logger=wandb_logger,
        # strategy=DDPSpawnPlugin(find_unused_parameters=False),
        strategy=DDPSpawnPlugin(find_unused_parameters=True),
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        num_sanity_val_steps=args.num_sanity_val_steps,
        callbacks=[LearningRateMonitor()],
    )
    print(f"dev {args.gpus}")
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    seed_everything(42)
    args = parser.parse_args()
    main(args)
    wandb.finish()
