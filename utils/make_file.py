#!/usr/bin/env python3
import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random

def make_dict_score_based(head_drop_ratio, csv_in_path):
    # 파일을 불러오는 부분
    data = pd.read_csv(csv_in_path)
    
    # 딕셔너리 생성 (Block, Head 튜플 사용)
    data_dict = {(int(row['Block']), int(row['Head'])): row['uAP'] for index, row in data.iterrows()}

    # 딕셔너리를 uAP 값에 따라 내름차순으로 정렬 => 그래야 변동없는 것부터 중요한 순으로 나열됨
    # order first = Fasle, order back = True
    sorted_data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
    # sorted_data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=False))

    # head_drop_ratio에 따라 상위 퍼센트의 항목을 추출
    num_items_to_select = int(len(sorted_data_dict) * head_drop_ratio)
    # num_items_to_select = int(len(sorted_data_dict) * 0.5)
    selected_items = dict(list(sorted_data_dict.items())[:num_items_to_select])
    print(selected_items)

    return selected_items


def make_dict_similarity_based(head_drop_ratio, csv_in_path):

    # 파일을 불러오는 부분
    data = pd.read_csv(csv_in_path)
    
    # 딕셔너리 생성 (Block, Head 튜플 사용)
    sim_dict = {(int(row['Block']), int(row['Head'])): row['Similarity'] for index, row in data.iterrows()}

    # 딕셔너리를 Similarity 값에 따라 내림차순으로 정렬 => 그래야 중복된 것부터 중복안되는 중요한 순으로 나열됨
    sorted_similarity_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
    # sorted_similarity_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=False))

    # head_drop_ratio에 따라 상위 퍼센트의 항목을 추출
    num_items_to_select = int(len(sorted_similarity_dict) * head_drop_ratio)
    # num_items_to_select = int(len(sorted_data_dict) * 0.5)
    selected_items = dict(list(sorted_similarity_dict.items())[:num_items_to_select])
    # print(selected_items)

    return selected_items

def make_dict_random_based(head_drop_ratio, csv_in_path):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    # 파일을 불러오는 부분
    data = pd.read_csv(csv_in_path)
    
    # 딕셔너리 생성 (Block, Head 튜플 사용)
    sim_dict = {(int(row['Block']), int(row['Head'])): row['Similarity'] for index, row in data.iterrows()}

    # 딕셔너리의 항목들을 랜덤으로 섞음
    items = list(sim_dict.items())
    random.shuffle(items)
    shuffled_similarity_dict = dict(items)

    # head_drop_ratio에 따라 상위 퍼센트의 항목을 추출
    num_items_to_select = int(len(shuffled_similarity_dict) * head_drop_ratio)
    selected_items = dict(list(shuffled_similarity_dict.items())[:num_items_to_select])
    # print(selected_items)

    return selected_items

# def make_hist_seperate(csv_in_path, hist_out_path):
#     data = pd.read_csv(csv_in_path)

#     # Ratio vs. uAP 그래프
#     plt.figure(figsize=(10, 6))
#     plt.plot(data['Ratio'], data['uAP'], marker='o', color='b')
#     plt.title('Ratio vs. uAP')
#     plt.xlabel('Ratio')
#     plt.ylabel('uAP')
#     plt.grid(True)
#     plt.savefig(hist_out_path + '/ratio_vs_uAP.png')
#     plt.close()
    
#     # Ratio vs. accuracy-at-1 그래프
#     plt.figure(figsize=(10, 6))
#     plt.plot(data['Ratio'], data['accuracy-at-1'], marker='o', color='g')
#     plt.title('Ratio vs. accuracy-at-1')
#     plt.xlabel('Ratio')
#     plt.ylabel('accuracy-at-1')
#     plt.grid(True)
#     plt.savefig(hist_out_path + '/ratio_vs_accuracy_at_1.png')
#     plt.close()
    
#     # Ratio vs. recall-at-p90 그래프
#     plt.figure(figsize=(10, 6))
#     plt.plot(data['Ratio'], data['recall-at-p90'], marker='o', color='r')
#     plt.title('Ratio vs. recall-at-p90')
#     plt.xlabel('Ratio')
#     plt.ylabel('recall-at-p90')
#     plt.grid(True)
#     plt.savefig(hist_out_path + '/ratio_vs_recall_at_p90.png')
#     plt.close()

def make_hist(csv_in_path, hist_out_path):
    data = pd.read_csv(csv_in_path)

    # 첫 번째 그래프: Ratio에 대한 여러 메트릭 비교
    plt.figure(figsize=(10, 6))
    plt.plot(data['Ratio'], data['uAP'], marker='o', color='b', label='uAP')
    plt.plot(data['Ratio'], data['accuracy-at-1'], marker='x', color='g', label='accuracy-at-1')
    # plt.plot(data['Ratio'], data['recall-at-p90'], marker='^', color='r', label='recall-at-p90')
    plt.title('Metrics vs. Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('Metrics')
    plt.grid(True)
    plt.legend()
    if args.based == 'score':
        plt.savefig(hist_out_path + '/metrics_comparison_score_prft.png')
    elif args.based == 'similarity':
        plt.savefig(hist_out_path + '/metrics_comparison_similarity_based_ascending.png')
    else:
        plt.savefig(hist_out_path + '/metrics_comparison_random_based.png')
    plt.close()

    # # 두 번째 그래프: GMAC vs. uAP
    # plt.figure(figsize=(10, 6))
    # plt.plot(data['Ratio'], data['Flops'], marker='o', color='orange')
    # plt.title('GMAC vs. Ratio')
    # plt.xlabel('Ratio')
    # plt.ylabel('GMAC')
    # plt.grid(True)
    # if args.based == 'score':
    #     plt.savefig(hist_out_path + '/GMAC_score_based.png')
    # elif args.based == 'similarity':
    #     plt.savefig(hist_out_path + '/GMAC_similarity_based_ascending.png')
    # else:
    #     plt.savefig(hist_out_path + '/GMAC_random_based.png')
    # plt.close()

      
def make_combined_hist(excel_path1, excel_path2, hist_out_path):
    data1 = pd.read_csv(excel_path1)
    data2 = pd.read_csv(excel_path2)

    # Ratio vs. uAP 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(data1['Ratio'], data1['uAP'], marker='o', color='b', label='Score-based uAP')
    plt.plot(data2['Ratio'], data2['uAP'], marker='o', color='b', linestyle='dashed', label='Sequence-based uAP')
    plt.title('Score-based vs. Sequence-based uAP')
    plt.xlabel('Ratio')
    plt.ylabel('uAP')
    plt.grid(True)
    plt.legend()
    plt.savefig(hist_out_path + '/Score_vs_Sequence_uAP.png')
    plt.close()
    
    # Ratio vs. accuracy-at-1 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(data1['Ratio'], data1['accuracy-at-1'], marker='o', color='g', label='Score-based accuracy-at-1')
    plt.plot(data2['Ratio'], data2['accuracy-at-1'], marker='o', color='g', linestyle='dashed', label='Sequence-based accuracy-at-1')
    plt.title('Score-based vs. Sequence-based accuracy-at-1')
    plt.xlabel('Ratio')
    plt.ylabel('accuracy-at-1')
    plt.grid(True)
    plt.legend()
    plt.savefig(hist_out_path + '/Score_vs_Sequence_accuracy_at_1.png')
    plt.close()
    
    # Ratio vs. recall-at-p90 그래프
    # plt.figure(figsize=(10, 6))
    # plt.plot(data1['Ratio'], data1['recall-at-p90'], marker='o', color='r', label='Fine-Tuned FP32 recall-at-p90')
    # plt.plot(data2['Ratio'], data2['recall-at-p90'], marker='o', color='r', linestyle='dashed', label='Fine-Tuned INT8 recall-at-p90')
    # plt.title('Fine-Tuned FP32 vs. INT8 recall-at-p90')
    # plt.xlabel('Ratio')
    # plt.ylabel('recall-at-p90')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(hist_out_path + '/Fine-Tuned fp32_vs_int8_recall_at_p90.png')
    # plt.close()

def make_combined_graph(excel_path1, excel_path2, graph_out_path):
    data1 = pd.read_csv(excel_path1)
    data2 = pd.read_csv(excel_path2)

    plt.figure(figsize=(12, 6))

    # uAP 그래프
    plt.plot(data1['Ratio'], data1['uAP'], marker='o', color='b', label='Data1 uAP (Solid)', linestyle='solid')
    plt.plot(data2['Ratio'], data2['uAP'], marker='o', color='b', label='Data2 uAP (Dashed)', linestyle='dashed')

    # accuracy-at-1 그래프
    plt.plot(data1['Ratio'], data1['accuracy-at-1'], marker='o', color='g', label='Data1 accuracy-at-1 (Solid)', linestyle='solid')
    plt.plot(data2['Ratio'], data2['accuracy-at-1'], marker='o', color='g', label='Data2 accuracy-at-1 (Dashed)', linestyle='dashed')

    plt.title('Combined Graph of uAP and accuracy-at-1')
    plt.xlabel('Ratio')
    plt.ylabel('Values')
    plt.grid(True)
    plt.legend()
    plt.savefig(graph_out_path + '/Score_vs_Sequence_accuracy_combined.png')
    plt.close()

def plot_line_graph_from_excel(file_path, x_column_index, y_column_index):
    # 데이터 로드
    df = pd.read_excel(file_path)

    # 선 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(df.iloc[:, x_column_index], df.iloc[:, y_column_index], marker='o', linestyle='-', color='blue')
    plt.xlabel(df.columns[x_column_index])
    plt.ylabel(df.columns[y_column_index])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv_in_path', type=str, help='Path to input CSV file')
    parser.add_argument('--csv_in_path2', type=str, help='Path to input CSV file')
    parser.add_argument('--hist_out_path', type=str, help='Path to output histogram images')
    parser.add_argument('--based', type=str, help='Pruning Mode score or similarity')

    args = parser.parse_args()
    # make_hist(args.csv_in_path, args.hist_out_path)
    make_combined_graph(args.csv_in_path, args.csv_in_path2, args.hist_out_path)