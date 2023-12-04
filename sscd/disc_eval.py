#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dataclasses
import json
import logging
import os
from typing import Optional

import faiss
import torch
import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from PIL import Image
# import cv2

from lib import initialize  # noqa
# from lib.inference_og import Inference
from lib.inference_sscd_og import Inference
from sscd.train_og import DISCData
from sscd.datasets.disc import DISCEvalDataset
from sscd.lib.util import parse_bool

parser = argparse.ArgumentParser()
inference_parser = parser.add_argument_group("Inference")
Inference.add_parser_args(inference_parser)

disc_parser = parser.add_argument_group("DISC")
disc_parser.add_argument("--disc_path", required=True)
disc_parser.add_argument(
    "--codecs",
    default=None,
    help="FAISS codecs for postprocessing embeddings as ';' separated strings "
    "in index_factory format",
)
disc_parser.add_argument(
    "--score_norm",
    default="1.0[0,2]",
    help="Score normalization settings, ';' separated, in format: "
    "<weight>[<first index>,<last index>]",
)
disc_parser.add_argument("--k", default=10, type=int)
# disc_parser.add_argument("--k", default=3, type=int)
disc_parser.add_argument(
    "--global_candidates",
    default=False,
    type=parse_bool,
    help="Use a global set of KNN candidates, instead of k per query. Uses CPU KNN.",
)
disc_parser.add_argument("--metadata", help="Metadata column to put in the result CSV")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("disc_eval.py")
logger.setLevel(logging.INFO)


class ProjectionError(Exception):
    """Projection returned non-finite values."""


def get_codecs(dims, is_l2_normalized, codecs_arg):
    if codecs_arg:
        return codecs_arg.split(";")
    if is_l2_normalized:
        return ["Flat", f"PCAW{dims},L2norm,Flat"]
    return ["Flat", "L2norm,Flat", f"PCAW{dims},L2norm,Flat", f"L2norm,PCAW{dims},Flat"]


def is_l2_normalized(embeddings):
    norms = linalg.norm(embeddings, axis=1)
    return np.abs(norms - 1).mean() < 0.01


@dataclasses.dataclass
class ScoreNormalization:
    weight: float
    start_index: int
    end_index: int

    @classmethod
    def parse(cls, spec):
        weight, spec = spec.split("[", 1)
        assert spec.endswith("]")
        spec = spec[:-1]
        if "," in spec:
            start, end = spec.split(",", 1)
        else:
            start = spec
            end = spec
        return cls(weight=float(weight), start_index=int(start), end_index=int(end))

    def __str__(self):
        return f"{self.weight:.2f}[{self.start_index},{self.end_index}]"

    __repr__ = __str__


@dataclasses.dataclass
class Embeddings:
    ids: np.ndarray
    embeddings: np.ndarray

    @property
    def size(self):
        return self.embeddings.shape[0]

    @property
    def dims(self):
        return self.embeddings.shape[1]

    def project(self, codec_index, codec_str) -> "Embeddings":
        projected = codec_index.sa_encode(self.embeddings)
        projected = np.frombuffer(projected, dtype=np.float32).reshape(self.size, -1)
        if not np.isfinite(projected).all():
            raise ProjectionError(
                f"Projection to {codec_str} resulted in non-finite values"
            )
        return dataclasses.replace(self, embeddings=projected)


def dataset_split(outputs, split_id) -> Embeddings:
    split = outputs["split"]
    this_split = split == split_id
    embeddings = outputs["embeddings"][this_split, :]
    image_num = outputs["image_num"][this_split]
    order = np.argsort(image_num)
    embeddings = embeddings[order, :]
    image_num = image_num[order]
    return Embeddings(ids=image_num, embeddings=embeddings)


def evaluate_all(dataset, outputs, codecs_arg, score_norm_arg, **kwargs):
    embeddings = outputs["embeddings"]
    
    ##############################################################################################################################
    # MODI for visualize attention map
    ##############################################################################################################################

    # # 시각화 코드 추가 => train 시에는 닫아두기
    # query_path = '/hdd/wi/dataset/DISC2021_mini/queries/images/queries/'
    # ref_path = '/hdd/wi/dataset/DISC2021_mini/references/images/references/'

    # # 쿼리 이미지 시각화
    # visualize_all_attention_maps_grid(query_path, queries.ids, outputs, "Query")
    # # 참조 이미지 시각화
    # visualize_all_attention_maps_grid(ref_path, refs.ids, outputs, "Ref")
    ##############################################################################################################################
    
    codecs = get_codecs(embeddings.shape[1], is_l2_normalized(embeddings), codecs_arg)
    logger.info("Using codecs: %s", codecs)
    score_norms = [None]
    if score_norm_arg:
        score_norms.extend(
            [ScoreNormalization.parse(spec) for spec in score_norm_arg.split(";")]
        )
    logger.info("Using score_norm: %s", score_norms)
    queries = dataset_split(outputs, DISCEvalDataset.SPLIT_QUERY)
    refs = dataset_split(outputs, DISCEvalDataset.SPLIT_REF)
    training = dataset_split(outputs, DISCEvalDataset.SPLIT_TRAIN)
    logger.info(
        "Dataset size: %d query, %d ref, %d train",
        queries.size,
        refs.size,
        training.size,
    )
    all_metrics = []
    for score_norm in score_norms:
        for codec in codecs:
            record = dict(codec=codec, score_norm=str(score_norm))
            metrics = evaluate(
                dataset, queries, refs, training, score_norm, codec, **kwargs
            )
            if metrics:
                record.update(metrics)
                all_metrics.append(record)
    return all_metrics

# def visualize_attention_map_for_set(image_path, image_ids, outputs, set_name):
#     for i, image_id in enumerate(image_ids):
#         idx = np.where(outputs['image_num'] == image_id)[0][0]
        
#         # 이미지 파일 이름 형식에 맞춰 수정
#         if set_name == "Query":
#             formatted_image_id = f"Q{image_id:05d}"  # 쿼리 이미지: Q + 5자리 숫자
#         elif set_name == "Ref":
#             formatted_image_id = f"R{image_id:06d}"  # 참조 이미지: R + 6자리 숫자
        
#         # 이미지 경로 구성 및 로드
#         img_path = os.path.join(image_path, f"{formatted_image_id}.jpg")
#         img = Image.open(img_path).resize((224, 224))

#         # 어텐션 맵 추출 및 리사이즈
#         cls_attn = outputs['attention_map'][idx][1:]
#         cls_attn = cls_attn.reshape((14, 14))
#         cls_attn = cv2.resize(cls_attn, (224, 224))
#         cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())  # 정규화

#         # 어텐션 맵을 컬러맵으로 변환
#         cls_attn_colored = plt.get_cmap('jet')(cls_attn)[:, :, :3]
#         cls_attn_colored = (cls_attn_colored * 255).astype(np.uint8)

#         # 원본 이미지와 어텐션 맵 오버레이
#         overlayed_img = cv2.addWeighted(np.array(img), 1, cls_attn_colored, 0.6, 0)

#         # 결과 저장
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#         ax[0].imshow(img)
#         ax[0].set_title(f"{set_name} Image")
#         ax[1].imshow(overlayed_img)
#         ax[1].set_title(f"{set_name} Attention Map Overlay")
#         plt.savefig(f"/hdd/wi/sscd-copy-detection/result/atten_map/{set_name}_{formatted_image_id}.png")
#         plt.close(fig)


# def visualize_all_attention_maps_grid(image_path, image_id, outputs, set_name):
    
#     for id in image_id:  # image_ids 배열의 각 원소에 대해 반복
#         formatted_image_id = f"{set_name[0]}{id:05d}" if set_name == "Query" else f"{set_name[0]}{id:06d}"

#     # 어텐션 맵의 블록과 헤드 수 확인
#     num_blocks = len(outputs['attention_maps'])
#     num_heads = outputs['attention_maps'][0].shape[0]  # 첫 번째 블록의 헤드 수를 기준으로 함

#     # Define your custom color map
#     # Replace the color values with the ones you extract from your screenshot
#     colors = ['#000000', '#5741D9', '#FDFDFD', '#FD4D4D']  # Example HEX colors
#     cmap_name = 'custom_cmap'
#     n_bins = 100  # Number of color bins
#     custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

#     # 그리드 형태로 어텐션 맵 시각화
#     fig, axes = plt.subplots(num_heads, num_blocks, figsize=(num_blocks * 2, num_heads * 2))
#     for i in range(num_blocks):
#         for j in range(num_heads):
#             attn_map = outputs['attention_maps'][i][j].reshape((14, 14))
#             attn_map = cv2.resize(attn_map, (224, 224))
#             attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())  # 정규화
#             axes[j, i].imshow(attn_map, cmap='jet')
#             axes[j, i].axis('off')

#     plt.suptitle(f"{set_name} Attention Maps for {formatted_image_id}")
#     plt.savefig(f"/hdd/wi/sscd-copy-detection/result/atten_map/{set_name}_AllAttn_{formatted_image_id}.png")
#     plt.close(fig)

def project(
    codec_str: str, queries: Embeddings, refs: Embeddings, training: Embeddings
):
    if codec_str != "Flat":
        assert codec_str.endswith(",Flat")
        codec = faiss.index_factory(training.dims, codec_str)
        # print(training.embeddings)
        codec.train(training.embeddings)
        queries = queries.project(codec, codec_str)
        refs = refs.project(codec, codec_str)
        training = training.project(codec, codec_str)
    return queries, refs, training


def evaluate(
    dataset: DISCEvalDataset,
    queries: Embeddings,
    refs: Embeddings,
    training: Embeddings,
    score_norm: Optional[ScoreNormalization],
    codec,
    **kwargs,
):
    try:
        queries, refs, training = project(codec, queries, refs, training)
    except ProjectionError as e:
        logger.error(f"DISC eval {codec}: {e}")
        return None
    eval_kwargs = dict(kwargs)
    use_gpu = torch.cuda.is_available()
    if score_norm:
        queries, refs = apply_score_norm(
            queries, refs, training, score_norm, use_gpu=use_gpu
        )
        eval_kwargs["metric"] = faiss.METRIC_INNER_PRODUCT
    metrics = dataset.retrieval_eval_splits(
        queries.ids,
        queries.embeddings,
        refs.ids,
        refs.embeddings,
        use_gpu=use_gpu,
        **eval_kwargs,
    )
    logger.info(
        f"DISC eval ({score_norm or 'no norm'}, {codec}): {json.dumps(metrics)}"
    )
    
    return metrics
    
def apply_score_norm(
    queries, refs, training, score_norm: ScoreNormalization, use_gpu=False
):
    index = faiss.IndexFlatIP(training.dims)
    index.add(training.embeddings)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    D, I = index.search(queries.embeddings, score_norm.end_index + 1)
    adjustment = -score_norm.weight * np.mean(
        D[:, score_norm.start_index : score_norm.end_index + 1],
        axis=1,
        keepdims=True,
    )
    ones = np.ones_like(refs.embeddings[:, :1])
    adjusted_queries = np.concatenate([queries.embeddings, adjustment], axis=1)
    adjusted_refs = np.concatenate([refs.embeddings, ones], axis=1)
    queries = dataclasses.replace(queries, embeddings=adjusted_queries)
    refs = dataclasses.replace(refs, embeddings=adjusted_refs)
    return queries, refs


def main(args):
    logger.info("Setting up dataset")
    dataset = DISCData.make_validation_dataset(
        args.disc_path,
        size=args.size,
        # include_train = False, # true면 score norm 에러남
        include_train=True,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
    )
    outputs = Inference.inference(args, dataset)
    logger.info("Retrieval eval")
    eval_options = dict(k=args.k, global_candidates=args.global_candidates)
    records = evaluate_all(
        dataset, outputs, args.codecs, args.score_norm, **eval_options
    )
    df = pd.DataFrame(records)
    if args.metadata:
        df["metadata"] = args.metadata
    csv_filename = os.path.join(args.output_path, "disc_metrics.csv")
    df.to_csv(csv_filename, index=False)
    with open(csv_filename, "r") as f:
        logger.info("DISC metrics:\n%s", f.read())


    
if __name__ == "__main__":
  
    args = parser.parse_args()
    main(args)
