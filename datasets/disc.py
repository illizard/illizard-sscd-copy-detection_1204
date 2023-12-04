# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from typing import Callable, Dict, Optional
from torchvision.datasets.folder import default_loader

from sscd.datasets.image_folder import get_image_paths
from sscd.datasets.isc.descriptor_matching import (
    knn_match_and_make_predictions,
    match_and_make_predictions,
)
from sscd.datasets.isc.io import read_ground_truth
from sscd.datasets.isc.metrics import evaluate, Metrics

# MODI for visualize
import matplotlib.pyplot as plt
import os



class DISCEvalDataset:
    """DISC2021 evaluation dataset."""

    SPLIT_REF = 0
    SPLIT_QUERY = 1
    SPLIT_TRAIN = 2
    # MODI
    k = 10
    
    def __init__(
        self,
        path: str,
        transform: Callable = None,
        include_train: bool = True,
        # Specific paths for each part of the dataset. If not set, inferred from `path`.
        query_path: Optional[str] = None,
        ref_path: Optional[str] = None,
        train_path: Optional[str] = None,
        gt_path: Optional[str] = None,
    ):
        def get_path(override_path, relative_path):
            return override_path if override_path else os.path.join(path, relative_path)

        query_path = get_path(query_path, "images/queries")
        ref_path = get_path(ref_path, "images/references_withD_20k")
        train_path = get_path(train_path, "images/train_20k") if include_train else None

        # query_path = get_path(query_path, "queries/images/queries")
        # ref_path = get_path(ref_path, "references/images/references")
        # train_path = get_path(train_path, "train_10k/images/train") if include_train else None

        gt_path = get_path(gt_path, "list_files/dev_ground_truth.csv")
        self.files, self.metadata = self.read_files(ref_path, self.SPLIT_REF)
        query_files, query_metadata = self.read_files(query_path, self.SPLIT_QUERY)
        self.files.extend(query_files)
        self.metadata.extend(query_metadata)
        if train_path:
            train_files, train_metadata = self.read_files(train_path, self.SPLIT_TRAIN)
            self.files.extend(train_files)
            self.metadata.extend(train_metadata)
        self.gt = read_ground_truth(gt_path)
        self.transform = transform

        # MODI VISUAL 추가된 부분
        self.query_path = query_path
        self.ref_path = ref_path

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        img = default_loader(filename)
        if self.transform:
            img = self.transform(img)
        sample = {"input": img, "instance_id": idx}
        sample.update(self.metadata[idx])
        return sample

    def __len__(self):
        return len(self.files)

    @classmethod
    def read_files(cls, path, split):
        files = get_image_paths(path)
        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        metadata = [
            dict(name=name, split=split, image_num=int(name[1:]), target=-1)
            for name in names
        ]
        return files, metadata

    def retrieval_eval(
        self, embedding_array, targets, split, **kwargs
    ) -> Dict[str, float]:
        query_mask = split == self.SPLIT_QUERY
        ref_mask = split == self.SPLIT_REF
        query_ids = targets[query_mask]
        query_embeddings = embedding_array[query_mask, :]
        ref_ids = targets[ref_mask]
        ref_embeddings = embedding_array[ref_mask, :]
        return self.retrieval_eval_splits(
            query_ids, query_embeddings, ref_ids, ref_embeddings, **kwargs
        )

    def retrieval_eval_splits(
        self,
        query_ids,
        query_embeddings,
        ref_ids,
        ref_embeddings,
        use_gpu=False,
        #MODI
        # k=10,
        k=k,
        global_candidates=False,
        **kwargs
    ) -> Dict[str, float]:
        query_names = ["Q%05d" % i for i in query_ids]
        ref_names = ["R%06d" % i for i in ref_ids]
        if global_candidates:
            predictions = match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                num_results=k * len(query_names),
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        else: # 시각화 하려면 knn 함수 리턴 한 것중에 이미지 이름 돌려주면 됨
            predictions = knn_match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                k=k,
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        results: Metrics = evaluate(self.gt, predictions)
        ##################################
        # MODI for visualize
        ##################################
        # 시각화 코드 추가 => train 시에는 닫아두기
        # self.visualize_predictions(predictions, top_k=k)
    
        return {
            "uAP": results.average_precision,
            "accuracy-at-1": results.recall_at_rank1,
            "recall-at-p90": results.recall_at_p90 or 0.0,
        }

    ##################################
    # MODI for visualize
    ##################################
    
    def visualize_predictions(self, predictions, top_k=k):
        # 이미지 로드 함수
        def load_image(folder, image_name):
            return default_loader(os.path.join(folder, image_name + '.jpg'))  # .jpg 확장자를 가정합니다.
        
        # 쿼리별로 그룹화
        grouped_predictions = {}
        for p in predictions:
            if p.query not in grouped_predictions:
                grouped_predictions[p.query] = []
            grouped_predictions[p.query].append(p)
        
        for query, preds in grouped_predictions.items():
            # 상위 k개의 이미지만 선택
            top_preds = sorted(preds, key=lambda x: x.score, reverse=True)[:5]
            
            # 시각화
            plt.figure(figsize=(15, 5))
            # 쿼리 이미지 표시
            plt.subplot(1, top_k + 2, 1)  # +2는 쿼리와 참조 이미지들 사이의 빈 칸과 끝 공백을 위함
            plt.imshow(load_image(self.query_path, query))
            plt.title(f"Query: {query}")
            plt.axis('off')
            
            # 상위 k개의 검색 결과 표시
            for i, pred in enumerate(top_preds, 3):  # 시작 위치를 3으로 설정
                plt.subplot(1, top_k + 2, i)
                plt.imshow(load_image(self.ref_path, pred.db))
                plt.title(f"Score: {pred.score:.2f}")
                plt.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join('/hdd/wi/sscd-copy-detection/result/pred_vis', f"{query}_predictions.png")
            plt.savefig(save_path)
            plt.close()
