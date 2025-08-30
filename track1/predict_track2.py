import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
import laion_clap
from torch.utils.data import DataLoader
from tqdm import tqdm

from mos_track2 import MosPredictor, MyDataset

from utils import *


def systemID(wavID):
    # e.g. audiomos2025-track2-sys0v1wm-utt0dJodp → sys0v1wm
    return wavID.replace("audiomos2025-track2-", "").split('-')[0]

def read_truth_jsonl(filepath):
    """
    Returns: dict[wavID] = [avg1, avg2, avg3, avg4]
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            wavID = item["data_path"]
            avg_scores = [
                np.mean(item["Production_Quality"]),
                np.mean(item["Production_Complexity"]),
                np.mean(item["Content_Enjoyment"]),
                np.mean(item["Content_Usefulness"])
            ]
            data[wavID] = avg_scores
    return data

def aggregate_by_system(data_dict):
    """
    Returns: dict[sysID] = [avg1, avg2, avg3, avg4]
    """
    sys_scores = {}
    for wavID, scores in data_dict.items():
        sid = systemID(wavID)
        sys_scores.setdefault(sid, []).append(scores)

    sys_agg = {}
    for sid, scores_list in sys_scores.items():
        scores_arr = np.array(scores_list)
        avg_scores = np.mean(scores_arr, axis=0)
        sys_agg[sid] = avg_scores.tolist()
    return sys_agg

def eval_system_metrics(truth_dict, pred_dict):
    truth_agg = aggregate_by_system(truth_dict)
    pred_agg  = aggregate_by_system(pred_dict)

    keys = sorted(set(truth_agg.keys()) & set(pred_agg.keys()))
    truth_scores = np.array([truth_agg[k] for k in keys])
    pred_scores = np.array([pred_agg[k] for k in keys])

    print("\n========== SYSTEM LEVEL ==========")
    labels = ["Production_Quality", "Production_Complexity", "Content_Enjoyment", "Content_Usefulness"]

    for i in range(4):
        t = truth_scores[:, i]
        p = pred_scores[:, i]
        mse = np.mean((t - p) ** 2)
        lcc = np.corrcoef(t, p)[0, 1]
        srcc = scipy.stats.spearmanr(t, p)[0]
        ktau = scipy.stats.kendalltau(t, p)[0]
        print(f"---- {labels[i]} ----")
        print(f"MSE  : {mse:.4f}")
        print(f"LCC  : {lcc:.4f}")
        print(f"SRCC : {srcc:.4f}")
        print(f"KTAU : {ktau:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=False,
                        default='../data')
    parser.add_argument('--ckptpath', type=str, required=True,
                        help='finetuned ckpt 檔案路徑')
    parser.add_argument('--expname', type=str, required=True,
                        help='輸出資料夾')
    parser.add_argument('--pooling_type', type=str, required=True, 
                        default='tsp', choices=['tap','tsp','sap','asp','gsap','gasp','drap', 'drsp'], help='Pooling type')
    parser.add_argument('--segment_size', type=int, required=False, 
                        default=1, help='Number of frames per segment in gsap, gasp, drap, drsp')
    parser.add_argument('--segment_sizes', nargs='+', type=int, default=[4,8,16,32], help='List of segment sizes for multi-scale pooling')

    args = parser.parse_args()

    os.makedirs(args.expname, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # upstream & model
    model = laion_clap.CLAP_Module(enable_fusion=False,
                                   device=device,
                                   amodel='HTSAT-base')

    net = MosPredictor(model, 512,
                    pooling_type=args.pooling_type,
                    segment_size=args.segment_size,
                    multi_segment_sizes=args.segment_sizes,
                    ).to(device)
    net.eval()
    # ckpt = torch.load(args.ckptpath, map_location=device)
    ckpt = torch.load(args.ckptpath, map_location=lambda storage, loc: storage.cuda() if device.type == 'cuda' else storage)
    net.load_state_dict(ckpt)

    # data loader
    wavdir    = os.path.join(args.datadir, 'track2_data_eval/wav')
    test_csv = os.path.join(args.datadir, 'audiomos2025_track2_human_annotations.csv')
    test_jsonl = os.path.join(args.datadir, 'audiomos2025_track2_human_annotations.jsonl')
    test_set  = MyDataset(wavdir, test_csv)
    loader    = DataLoader(test_set,
                           batch_size=1,
                           shuffle=False,
                           collate_fn=test_set.collate_fn)

    l1 = nn.L1Loss()
    preds_pq = {}
    preds_pc = {}
    preds_ce = {}
    preds_cu = {}
    print("Start prediction...")
    with torch.no_grad():
        for wav, pq, pc, ce, cu, names in tqdm(loader):
            wav = wav.squeeze(1).to(device)

            out_pq, out_pc, out_ce, out_cu = net(wav)
            p_pq = out_pq.cpu().item()
            p_pc = out_pc.cpu().item()
            p_ce = out_ce.cpu().item()
            p_cu = out_cu.cpu().item()
            utt = names[0]
            preds_pq[utt] = p_pq
            preds_pc[utt] = p_pc
            preds_ce[utt] = p_ce
            preds_cu[utt] = p_cu

    # 存檔
    answer_file = os.path.join(args.expname, 'answer_track2.txt')
    with open(answer_file, 'w') as f:
        for utt in sorted(preds_pq):
            f.write(f"{utt},{preds_pq[utt]:.6f},{preds_pc[utt]:.6f},{preds_ce[utt]:.6f},{preds_cu[utt]:.6f}\n")

    # 讀 ground-truth
    truth = read_truth_jsonl(test_jsonl)
    pred  = {utt: (preds_pq[utt], preds_pc[utt], preds_ce[utt], preds_cu[utt]) for utt in preds_pq}

    # Eval
    eval_system_metrics(truth, pred)

if __name__ == '__main__':
    main()
