#!/usr/bin/env python
# coding: utf-8

# # [Fine-grained Incident Video Retrieval (FIVR)](https://arxiv.org/abs/1809.04094)
# ## Dataset
# - 数据集特征目录：/home/camp/FIVR/features/vcms_v1
# - Annotation目录：/home/camp/FIVR/annotation
# - 描述：我们提供视频的帧级别特征
#     - 每个h5文件有两个group: images和names
#     - images group 保存了每个视频帧的特征，id是视频id，value是帧特征
#     - names group 保存了每个视频帧的名字，id是视频id，value是帧的名字，例如\[1.jpg, 2.jpg,...\]
# - 一些关键词的解释
#     - vid: 和视频一一对应。
#     - name: 和视频一一对应，annotation中的使用的是name
#     - 通过vid2name和name2vid可以确定他们之间的映射关系
# - 三种相似的视频
#     - Duplicate Scene Video (DSV)
#     - Complementary Scene Video (CSV), 如果A，B两个视频描述的同一个事件，且时间上有overlap，则认为是彼此之间的相似关系是CSV
#     - Incident Scene Video (ISV)，如果A，B两个视频描述的是同一个时间，时间上没有overlap，则认为彼此之间的相似关系是ISV
# - 三个任务
#     - DSVR：负责检索出DSV的相似
#     - CSVR：负责检索出DSV+CSV的相似
#     - ISVR：负责检索出DSV+CSV+ISV的相似

import numpy as np
import os
import h5py
from tqdm import tqdm
from glob import glob
import pickle as pk
import json
import time
from scipy.spatial.distance import cdist
from future.utils import viewitems, lrange
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
import pdb
import time
import faiss
import collections

def read_h5file(path):
    hf = h5py.File(path, 'r')
    g1 = hf.get('images')
    g2 = hf.get('names')
    return g1.keys(), g1, g2

def load_features(dataset_dir, is_gv=True):
    '''
    加载特征
    :param dataset_dir: 特征所在的目录, 例如：/home/camp/FIVR/features/vcms_v1
    :param is_gv: 是否取平均。True：返回帧平均的结果，False：保留所有帧的特征
    :return:
    '''
    h5_paths = glob(os.path.join(dataset_dir, '*.h5'))
    print(h5_paths)
    vid2features = {}
    final_vids = []
    features = []
    for h5_path in h5_paths:
        vids, g1, g2 = read_h5file(h5_path)
        for vid in tqdm(vids):
            if is_gv:
                cur_arr = g1.get(vid)
                cur_arr = np.mean(cur_arr, axis=0, keepdims=False)
                cur_arr /= (np.linalg.norm(cur_arr, ord=2, axis=0))
                vid2features[vid] = cur_arr
            else:
                cur_arr = np.asarray(g1.get(vid))
                mean_arr = np.mean(cur_arr, axis=0, keepdims=True)
                cur_arr = np.concatenate([cur_arr, mean_arr], axis=0)
                vid2features[vid] = cur_arr
                final_vids.extend([vid] * len(cur_arr))
                features.extend(cur_arr)
    if is_gv:
        return vid2features
    else:
        return final_vids, features, vid2features

def evaluateOfficial(annotations, results, relevant_labels, dataset, quiet):
    """
      Calculate of mAP and interpolated PR-curve based on the FIVR evaluation process.
      Args:
        annotations: the annotation labels for each query
        results: the similarities of each query with the videos in the dataset
        relevant_labels: labels that are considered positives
        dataset: video ids contained in the dataset
      Returns:
        mAP: the mean Average Precision
        ps_curve: the values of the PR-curve
    """
    pr, mAP = [], []
    iterations = viewitems(annotations) if not quiet else tqdm(viewitems(annotations))
    for query, gt_sets in iterations:
        query = str(query)
        if query not in results: print('WARNING: Query {} is missing from the result file'.format(query)); continue
        if query not in dataset: print('WARNING: Query {} is not in the dataset'.format(query)); continue

        # set of relevant videos
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt: print('WARNING: Empty annotation set for query {}'.format(query)); continue

        # calculation of mean Average Precision (Eq. 6)
        i, ri, s = 0.0, 0, 0.0
        y_target, y_score = [], []
        for video, sim in sorted(viewitems(results[query]), key=lambda x: x[1], reverse=True):
            if video in dataset:
                y_score.append(sim)
                y_target.append(1.0 if video in query_gt else 0.0)
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        mAP.append(s / len(query_gt))
        if not quiet:
            print('Query:{}\t\tAP={:.4f}'.format(query, s / len(query_gt)))

        # add the dataset videos that are missing from the result file
        missing = len(query_gt) - y_target.count(1)
        y_target += [1.0 for _ in lrange(missing)] # add 1. for the relevant videos
        y_target += [0.0 for _ in lrange(len(dataset) - len(y_target))] # add 0. for the irrelevant videos
        y_score += [0.0 for _ in lrange(len(dataset) - len(y_score))]

        # calculation of interpolate PR-curve (Eq. 5)
        precision, recall, thresholds = precision_recall_curve(y_target, y_score)
        p = []
        for i in lrange(20, -1, -1):
            idx = np.where((recall >= i * 0.05))[0]
            p.append(np.max(precision[idx]))
        pr.append(p)
    # return mAP
    return mAP, np.mean(pr, axis=0)[::-1]

class GTOBJ:
    def __init__(self):
        annotation_path = '/home/camp/FIVR/annotation/annotation.json'
        dataset_path = '/home/camp/FIVR/annotation/youtube_ids.txt'
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        self.dataset = set(np.loadtxt(dataset_path, dtype=str).tolist())


if __name__ == '__main__':

    gtobj = GTOBJ()
    relevant_labels_mapping = {
        'DSVR': ['ND','DS'],
        'CSVR': ['ND','DS','CS'],
        'ISVR': ['ND','DS','CS','IS'],
    }

    final_vids, features, vid2features = load_features('/home/camp/FIVR/features/vcms_v1', is_gv=False)

    final_vids = np.asarray(final_vids)
    features = np.asarray(features)

    # 使用faiss建立索引, 并转移到gpu上
    d = 512
    index = 0
    isTest = 1

    if 1 == isTest:
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_gpu(res, 0, index)

    else:
        index = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_all_gpus(index)

    index.add(features)

    # 加载特征
    vids = list(vid2features.keys())
    print(vids[:10])

    # 加载vid2name 和 name2vid
    with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
        vid2names = pk.load(pk_file)

    with open('/home/camp/FIVR/name2vid.pk', 'rb') as pk_file:
        name2vids = pk.load(pk_file)

    # 开始评估
    annotation_dir = '/home/camp/FIVR/annotation'
    names = np.asarray([vid2names[vid][0] for vid in vids])
    query_names = None
    results = None

    # 计算每个视频有多少帧
    vid2frameNum = collections.defaultdict(lambda: 0)
    for i in final_vids:
        vid2frameNum[i] += 1

    # 建立一个vids2idx的映射dict
    vid2idx = {}
    for i, vid in enumerate(vids):
        vid2idx[vid] = i

    ## 计算方法
    # 对于有n个帧查询视频的每个帧，在所有视频帧中搜索与其最相似的K个帧并获得相似度，其他帧默认与该帧相似度为0
    # 然后根据搜索结果对所有视频打分，对所有视频按照分数排序即可
    # 但由于faiss最多支持K=1024，所以我们为了使K能够更大
    # 分为了两轮，第一轮对于有n个帧查询视频的每个帧，在所有视频帧中搜索与其最相似的kNeighbor[0]个帧
    # 第二轮对于这n*kNeighbor[0]个帧，在所有视频帧中搜索与其最相似的kNeighbor[1]个帧
    # 这样对于查询视频的每个帧，我们都获得了与其最相似的kNeighbor[0]*kNeighbor[1]个帧
    kNeighbor = [5, 233]

    # 计算结果并进行评估
    for task_name in ['DSVR', 'CSVR', 'ISVR']:
        annotation_path = os.path.join(annotation_dir, task_name + '.json')
        with open(annotation_path, 'r') as annotation_file:
            json_obj = json.load(annotation_file)
        if results is None:
            results = dict()
            query_names = json_obj.keys()
            query_names = [str(query_name) for query_name in query_names]
            query_vids = [name2vids[query_name] for query_name in query_names]
            query_frame_indexs = []

            for query_vid in query_vids:
                tmp = np.where(final_vids == query_vid)
                if len(tmp) != 0 and len(tmp[0]) != 0:
                    query_frame_indexs = tmp[0]
                else:
                    print('skip query: ', query_vid)
                
                t1 = time.time()
                # 对查询视频每个帧搜索最相似的kNeighbor[0]个帧
                query_frame_features = np.squeeze(features[query_frame_indexs])
                preD, preI = index.search(query_frame_features, kNeighbor[0])
                for i in range(preI.shape[0]):
                    gallery_feats = features[preI[i]]
                    mean_gallery_feats = np.mean(gallery_feats, axis=0, keepdims=False)
                    query_frame_features[i] = mean_gallery_feats

                D2, I2 = index.search(query_frame_features, kNeighbor[1])
                similarities = np.stack((I2, D2), axis=-1)

                # 对所有视频打分并排序获得结果
                scorelog = collections.defaultdict(lambda: 0.0)
                for i in range(similarities.shape[0]):
                    for j in range(similarities.shape[1]):
                        vid_idx = vid2idx[final_vids[int(similarities[i][j][0])]]
                        scorelog[final_vids[int(similarities[i][j][0])]] += similarities[i][j][1]

                res = []
                for k, v in scorelog.items():
                    if v > 0:
                        nb_frame = vid2frameNum[k]
                        res.append([k, v / nb_frame])
                res.sort(key=lambda a: -a[1])
                res_id_set = set([i[0] for i in res])
                left_id_set = set(vids) - res_id_set
                query_name = vid2names[query_vid][0]
                query_result = {}
                for i in res:
                    name, sim = vid2names[i[0]][0], i[1]
                    query_result[name] = sim
                for j in left_id_set:
                    name = vid2names[j][0]
                    query_result[name] = 0.0
                del query_result[query_name]
                results[query_name] = query_result
                t2 = time.time()
                print(t2 - t1)

        mAPOffcial, precisions = evaluateOfficial(annotations=gtobj.annotations, results=results,
                                                  relevant_labels=relevant_labels_mapping[task_name],
                                                  dataset=gtobj.dataset,
                                                  quiet=False)
        print('{} mAPOffcial is {}'.format(task_name, np.mean(mAPOffcial)))

