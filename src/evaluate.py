import copy
import math
import os
import random
from pathlib import Path

import numpy as np
from tqdm import trange
import argparse
import pickle
import time

import jittor as jt
import jittor.optim as optim
from types import SimpleNamespace
import numpy as np

import matplotlib.pyplot as plt

from shapely.geometry import Polygon as ShapelyPolygon

from .score_model import PolygonPackingTransformer

from .utils import SummaryWriter

from .utils import Polygon as PolygonGeometry
from .utils import setRandomSeed


from .sde import init_sde, lossFun, pc_sampler_state, ExponentialMovingAverage


from . import calutil
from . import rmspacing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"
POLYGON_DIR = DATA_DIR / "polygons"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

def cal_util(polys, pidsAll, actionsAll, eps=5.0):
    translationsAll = []
    thetasAll = []
    
    for i in range(len(pidsAll)):
        thetas = []
        translations = []
        for j in range(len(pidsAll[i])):
            thetas.append(float(math.atan2(actionsAll[i][j][3], actionsAll[i][j][2])))
            translations.append([float(actionsAll[i][j][0]), float(actionsAll[i][j][1])])
        thetasAll.append(thetas)
        translationsAll.append(translations)

    res = calutil.cal_util_all(pidsAll, thetasAll, copy.deepcopy(translationsAll), copy.deepcopy(polys), eps)

    # sumarea, maxinter, suminter, minx, miny, maxx, maxy
    res = np.array(res, dtype=np.float32)
    
    bbd_area = (res[:, 5] - res[:, 3]) * (res[:, 6] - res[:, 4])
    util = res[:, 0] / bbd_area
    valid = (res[:, 1] < eps)
    bbd = jt.stack([res[:, 3], res[:, 4], res[:, 5], res[:, 6]], dim=1)
    sum_inter_per = res[:, 2] / res[:, 0] * 100
    
    return util, valid, bbd, sum_inter_per

def existsOrMkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def readAllPolys(polyCnt=440):
    polyVertices = []
    for i in range(0, polyCnt):

        poly_path = POLYGON_DIR / f"{i}.txt"
        poly = PolygonGeometry(str(poly_path))
        maxContour = poly.getMaxContour()
        polyVertices.append(maxContour)
    
    return polyVertices

def genPaddingMask(polyVerticesData):
    paddingMaskData = []
    for i in range(len(polyVerticesData)):
        paddingMask = []
        for j in range(len(polyVerticesData[i])):
            paddingMask.append(0)
        #for j in range(len(polyVerticesData[i]), 50):
        #    paddingMask.append(-inf)
        paddingMaskData.append(paddingMask)
    return paddingMaskData

def padData(polyIds, actions):
    for i in range(len(polyIds)):
        for j in range(len(actions[i])):
            actions[i][j][2] *= 500
            actions[i][j][3] *= 500
        #for j in range(len(polyIds[i]), 50):
        #    polyIds[i].append(0)
        #    actions[i].append([0, 0, 0, 0])
    return polyIds, actions

def collectValiData():
    
    dirName = DATASET_DIR / "dataset_dental_sp7_r_h12_vali.pkl"
    dataFile = open(dirName, "rb")
    print("loading data from ", dirName)
    polyIds = pickle.load(dataFile)
    actions = pickle.load(dataFile)
    paddingMask = genPaddingMask(polyIds)
    polyIds, actions = padData(polyIds, actions)
    
    dataFile.close()

    
    return polyIds, actions, paddingMask


def removeSpacing(polys, pidsAll, actionsAll, height):
    translationsAll = []
    thetasAll = []
    
    for i in range(len(pidsAll)):
        thetas = []
        translations = []
        for j in range(len(pidsAll[i])):
            thetas.append(float(math.atan2(actionsAll[i][j][3], actionsAll[i][j][2])))
            translations.append([float(actionsAll[i][j][0]), float(actionsAll[i][j][1])])
        thetasAll.append(thetas)
        translationsAll.append(translations)

    rm_begin = time.time()
    rm_translations = rmspacing.rm_spacing_all(pidsAll, thetasAll, copy.deepcopy(translationsAll), copy.deepcopy(polys), 8000.0, height, 1.0)
    rm_end = time.time()
    
    newActionsAll = []
    for i in range(len(pidsAll)):
        newActions = []
        for j in range(len(pidsAll[i])):
            newActions.append([float(rm_translations[i][j][0]), float(rm_translations[i][j][1]), float(actionsAll[i][j][2]), float(actionsAll[i][j][3])])
        newActionsAll.append(newActions)
    
    return jt.float32(newActionsAll), rm_end - rm_begin



def calculate_angle(p1, p2, p3):
    a = [p1[0] - p2[0], p1[1] - p2[1]]
    b = [p3[0] - p2[0], p3[1] - p2[1]]
    dot_product = a[0] * b[0] + a[1] * b[1]
    cross_product = a[0] * b[1] - a[1] * b[0]
    angle = math.atan2(cross_product, dot_product)
    angle = abs(angle) * (180.0 / math.pi)
    if cross_product < 0:
        angle = 360 - angle
    return angle

def compute_node_features(poly):
    node_features = []
    
    for i in range(len(poly)):
        p1, p2, p3 = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
        # 计算内角
        internal_angle = calculate_angle(p1, p2, p3)
        node_features.append([p2[0], p2[1], internal_angle])

    return node_features

def compute_global_features(poly):
    polygon = ShapelyPolygon(poly)
    area = polygon.area
    perimeter = polygon.length

    return [area, perimeter]
        
def create_gnn_data(polygons):
    node_feats, edge_list, batch_idx, areas, perims = [], [], [], [], []
    node_offset = 0

    for g_idx, poly in enumerate(polygons):
        feats = jt.float32(compute_node_features(poly))              # [n, 3]
        num_nodes = feats.shape[0]

        edges = []
        for i in range(num_nodes):
            j = (i + 1) % num_nodes
            edges.append([node_offset + i, node_offset + j])
            edges.append([node_offset + j, node_offset + i])
        edge_list.append(jt.int32(edges))

        node_feats.append(feats)
        batch_idx.append(jt.full([num_nodes], g_idx, dtype='int32'))

        area, perm = compute_global_features(poly)
        areas.append(jt.float32([area]))
        perims.append(jt.float32([perm]))

        node_offset += num_nodes

    x = jt.concat(node_feats, dim=0)                                 # [N, 3]
    edge_index = jt.concat(edge_list, dim=0).transpose(1, 0)         # [2, E]
    batch = jt.concat(batch_idx, dim=0)                              # [N]
    area = jt.concat(areas, dim=0)                                   # [G, 1]
    perm = jt.concat(perims, dim=0)                                  # [G, 1]

    return SimpleNamespace(x=x, edge_index=edge_index, batch=batch, area=area, perm=perm)


def rotatePoly(poly, theta):
    # poly shape is 1, n, 2
    poly = poly.transpose(1, 2)

    cosTheta = jt.cos(theta)
    sinTheta = jt.sin(theta)
    rotationMatrices = jt.stack([cosTheta, -sinTheta, sinTheta, cosTheta], dim=1).view(-1, 2, 2)

    rotatedPoly = jt.bmm(rotationMatrices, poly)
    
    return rotatedPoly.squeeze(0).transpose(0, 1)

def visualize(epoch, polyIds, polyVertices, predict, paddingMask, writer, figName):
    
    plt.figure()
    for i, polyId in enumerate(polyIds):
        if paddingMask[i] < 0: continue
        polyVertex = polyVertices[polyId]
        polyTensor = jt.float32([polyVertex]).clone()

        theta = jt.atan2(predict[i, 3], predict[i, 2]).unsqueeze(0)
        rotatedPoly = rotatePoly(polyTensor, theta)
        
        res = predict[i, 0:2].reshape(1, 2) 
        newPoly = rotatedPoly + res
        newPoly = jt.cat((newPoly, newPoly[0].unsqueeze(0)), dim=0)
        plt.plot([float(p[0]) for p in newPoly], [float(p[1]) for p in newPoly])

    # plt.xlim(0, 2560)
    # plt.ylim(0, 7000)
    plt.title(figName + '_epoch_{}'.format(epoch))
    writer.add_figure(f"Images/epoch_{epoch}_{figName}", plt.gcf())
    plt.clf()
    plt.close()


def vali_res(valiPolyIds, polyVertices, valiActions, paddingMaskData, predict):
    
    polyList_toc = []
    polyIds_toc = []
    polyCnt = 0

    for i, polyId in enumerate(valiPolyIds):
        if paddingMaskData[i] < 0:
            continue
        polyList_toc.append(copy.deepcopy(polyVertices[polyId]))
        polyIds_toc.append(polyCnt)
        polyCnt += 1
    
        polyIdsAll_toc = []
        actionsAll_toc = []
        for i in range(len(predict)):
            polyIdsAll_toc.append(copy.deepcopy(polyIds_toc))
            actionsAll_toc.append(copy.deepcopy(predict[i, :polyCnt].tolist()))
        
        polyIdsAll_toc.append(copy.deepcopy(polyIds_toc))
        actionsAll_toc.append(copy.deepcopy(valiActions[:polyCnt].tolist()))
        
    util, valid, bbd, sum_inter_per = cal_util(copy.deepcopy(polyList_toc), polyIdsAll_toc, actionsAll_toc, 50.0)
    
    before_vali_cnt = 0
    before_vali_util_list = []
    before_intersec_area_list = []
    for i in range(len(util) - 1):
        before_intersec_area_list.append(sum_inter_per[i])
        if valid[i]:
            before_vali_cnt += 1
            before_vali_util_list.append(util[i])
    
    for i in range(len(predict)):
        predict[i, :polyCnt, 0] -= bbd[i, 0]
        predict[i, :polyCnt, 1] -= bbd[i, 1]
        actionsAll_toc[i] = predict[i, :polyCnt].tolist()
    
    rm_predict, rm_time = removeSpacing(copy.deepcopy(polyList_toc), polyIdsAll_toc, actionsAll_toc, 1205.0)

    rm_actionsAll_toc = []
    # contain gd
    for i in range(len(actionsAll_toc)):
        rm_actionsAll_toc.append(rm_predict[i, :polyCnt].tolist())
        
    rm_util, rm_valid, rm_bbd, rm_sum_inter_per = cal_util(copy.deepcopy(polyList_toc), polyIdsAll_toc, rm_actionsAll_toc, 50.0)
    
    intersec_area_list = []
    rm_vali_util_list = []
    
    rm_vali_cnt = 0
    
    best_util = 0
    best_id = 0
    for i in range(len(rm_util) - 1):
        intersec_area_list.append(rm_sum_inter_per[i])
        if rm_valid[i]:
            rm_vali_cnt += 1
            rm_vali_util_list.append(rm_util[i])
            if rm_util[i] > best_util:
                best_util = rm_util[i]
                best_id = i
                
    gt_util = util[-1]
    
    if len(rm_vali_util_list) == 0:
        rm_vali_util_list.append(0)
    if len(before_vali_util_list) == 0:
        before_vali_util_list.append(0)    
    
    return before_vali_cnt, rm_vali_cnt, gt_util, before_intersec_area_list, intersec_area_list, before_vali_util_list, rm_vali_util_list, rm_predict, best_id, rm_predict[-1], rm_time

if __name__ == '__main__':
    print("hi")
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--n', type=int, default=192)
    parser.add_argument('--m', type=int, default=192)
    parser.add_argument('--x', type=int, default=2000)
    parser.add_argument('--y', type=int, default=2000)
    
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--beginEpoch', type=int, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=25.)
    parser.add_argument('--sde_mode', type=str, default='ve')
    
    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--visualize_freq', type=int, default=64)
    parser.add_argument('--vali_freq', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=3407)
    # load args
    
    args = parser.parse_args()
    batchSize = args.batch_size
    beginEpoch = args.beginEpoch
    # n, m, x, y = args.n, args.m, args.x, args.y
    
    setRandomSeed(args.seed)
    
    jt.flags.use_cuda = 1

    existsOrMkdir(LOG_DIR)
    tb_path = LOG_DIR / args.log_dir / "eval"
    existsOrMkdir(tb_path)
    existsOrMkdir(CHECKPOINT_DIR)
    writer = SummaryWriter(str(tb_path))
    print("reading polys...")
    polyVertices = readAllPolys(440)
    gnnFeatureData = create_gnn_data(polyVertices)
    
    polyVertexNumbers = []
    for polyVertex in polyVertices:
        polyVertexNumbers.append(len(polyVertex))
        
    polyIdsVali, actionsVali, paddingMaskVali = collectValiData()
    polyIdsVali = polyIdsVali[:128]
    actionsVali = actionsVali[:128]
    paddingMaskVali = paddingMaskVali[:128]

    prior_fn, marginal_prob_fn, sde_fn, sampling_eps = init_sde(args.sde_mode)
    
    ''' Init Model '''
    
    score = PolygonPackingTransformer(marginal_prob_std_func=marginal_prob_fn)
    checkpoint_pickle_path = CHECKPOINT_DIR / "score_model.pkl"
    if checkpoint_pickle_path.exists():
        score.load(str(checkpoint_pickle_path))
        print(f"Loaded pretrained weights from {checkpoint_pickle_path}")
    else:
        print(f"No pretrained checkpoint found at {checkpoint_pickle_path}, initializing from scratch.")

                
    

    epoch = 0
    if True:
        choosedVali = random.randint(0, 128)
        valiPolyIds = jt.array(polyIdsVali[choosedVali],dtype=jt.int64)
        valiActions = jt.array(actionsVali[choosedVali],dtype=jt.float32)
        paddingMaskData = jt.array(paddingMaskVali[choosedVali],dtype=jt.float32)
        # valiHeighs = jt.array(heightsDataAll[choosedVali],dtype=jt.float32).squeeze(0)
        
        with jt.no_grad():
            samples, res = pc_sampler_state(score, sde_fn, len(valiPolyIds), valiPolyIds, gnnFeatureData, paddingMaskData)
        
        before_vali_cnt, rm_vali_cnt, gt_util, \
        before_intersec_area_list, intersec_area_list, \
        before_vali_util_list, rm_vali_util_list, \
        rm_predict, best_util_id, rm_vali_action,\
        rm_time = vali_res(valiPolyIds, polyVertices, valiActions, paddingMaskData, res)
        
        visualize(epoch, valiPolyIds, polyVertices, rm_vali_action, paddingMaskData, writer, "gd")
        visualize(epoch, valiPolyIds, polyVertices, rm_predict[best_util_id], paddingMaskData, writer, "pr")
        
        rm_best_util = max(rm_vali_util_list)
        rm_avg_util = sum(rm_vali_util_list) / len(rm_vali_util_list)
        rm_wrst_util = min(rm_vali_util_list)
        
        before_best_util = max(before_vali_util_list)
        before_avg_util = sum(before_vali_util_list) / len(before_vali_util_list)
        before_wrst_util = min(before_vali_util_list)
        
        before_best_inter = min(before_intersec_area_list)
        before_avg_inter = sum(before_intersec_area_list) / len(before_intersec_area_list)
        before_wrst_inter = max(before_intersec_area_list)
        
        print("total:",len(rm_vali_util_list),len(before_vali_util_list),len(before_intersec_area_list))
        print("before_best_inter=%f before_avg_inter=%f before_wrst_inter=%f" % (before_best_inter, before_avg_inter, before_wrst_inter))
        print("rm_best_util=%f rm_avg_util=%f rm_wrst_util=%f" % (rm_best_util, rm_avg_util, rm_wrst_util))
        print("before_vali_cnt=%d rm_vali_cnt=%d" % (before_vali_cnt, rm_vali_cnt))
        
        writer.add_scalars(
            'vali/util',
            {
                'PRBst': rm_best_util,
                'PRAvg': rm_avg_util,
                'PRWrst': rm_wrst_util,
                'GD': gt_util,
            },
            epoch,
        )
        writer.add_scalars(
            'vali/area',
            {
                'PRBst': before_best_inter,
                'PRAvg': before_avg_inter,
                'PRWrst': before_wrst_inter,
            },
            epoch,
        )
        writer.add_scalars(
            'vali/validCnt',
            {
                'bef': before_vali_cnt,
                'aft': rm_vali_cnt,
            },
            epoch,
        )


    writer.close()