import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
import laion_clap
from torch.utils.data import DataLoader
from tqdm import tqdm

from mos_track1 import MosPredictor, MyDataset

from utils import *


def systemID(wavID):
    return wavID.replace("audiomos2025-track1-","").split('_')[0]

def read_file(filepath):
    d = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3: 
                continue
            utt, o, t = parts
            d[utt] = (float(o), float(t))
    return d

def eval_metrics(truth_dict, pred_dict, label):
    keys = sorted(set(truth_dict.keys()) & set(pred_dict.keys()))
    truth_o = np.array([truth_dict[k][0] for k in keys])
    truth_t = np.array([truth_dict[k][1] for k in keys])
    pred_o  = np.array([pred_dict[k][0]  for k in keys])
    pred_t  = np.array([pred_dict[k][1]  for k in keys])

    print(f"\n========== {label.upper()} ==========")
    for name, t, p in [
        ("OVERALL", truth_o, pred_o),
        ("TEXTUAL", truth_t, pred_t),
    ]:
        mse  = np.mean((t - p)**2)
        lcc  = np.corrcoef(t, p)[0,1]
        srcc = scipy.stats.spearmanr(t, p)[0]
        ktau = scipy.stats.kendalltau(t, p)[0]
        print(f"---- {name} ----")
        print(f"MSE  : {mse:.4f}")
        print(f"LCC  : {lcc:.4f}")
        print(f"SRCC : {srcc:.4f}")
        print(f"KTAU : {ktau:.4f}")

def aggregate_by_system(d):
    so = {}
    st = {}
    for utt,(o,t) in d.items():
        sid = systemID(utt)
        so.setdefault(sid, []).append(o)
        st.setdefault(sid, []).append(t)
    return {k: np.mean(v) for k,v in so.items()}, {k: np.mean(v) for k,v in st.items()}

def eval_system_metrics(truth_dict, pred_dict):
    truth_o, truth_t = aggregate_by_system(truth_dict)
    pred_o,  pred_t  = aggregate_by_system(pred_dict)
    common = sorted(set(truth_o.keys()) & set(pred_o.keys()))
    truth_o_arr = np.array([truth_o[k] for k in common])
    pred_o_arr  = np.array([pred_o[k]  for k in common])
    truth_t_arr = np.array([truth_t[k] for k in common])
    pred_t_arr  = np.array([pred_t[k]  for k in common])

    print("\n========== SYSTEM LEVEL ==========")
    for name, t, p in [
        ("OVERALL", truth_o_arr, pred_o_arr),
        ("TEXTUAL", truth_t_arr, pred_t_arr),
    ]:
        mse  = np.mean((t - p)**2)
        lcc  = np.corrcoef(t, p)[0,1]
        srcc = scipy.stats.spearmanr(t, p)[0]
        ktau = scipy.stats.kendalltau(t, p)[0]
        print(f"---- {name} ----")
        print(f"MSE  : {mse:.4f}")
        print(f"LCC  : {lcc:.4f}")
        print(f"SRCC : {srcc:.4f}")
        print(f"KTAU : {ktau:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=False, default='../data/MusicEval-full', help='MusicEval-full 目錄')
    parser.add_argument('--ckptpath', type=str, required=True, help='finetuned ckpt 檔案路徑')
    parser.add_argument('--expname', type=str, required=True, help='輸出資料夾')
    parser.add_argument('--pooling_type', type=str, required=True, default='tsp', choices=['tap','tsp','sap','asp','gsap', 'gasp', 'drap', 'drsp'], help='Pooling type')
    parser.add_argument('--segment_size', type=int, required=False,  default=1, help='Number of frames per segment in gsap, gasp, mrap, mrsp')
    parser.add_argument('--segment_sizes', nargs='+', type=int, default=[4,8,16,32], help='List of segment sizes for multi-scale pooling')
    parser.add_argument('--num_heads', type=int, required=False, default=1)

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
                    num_heads=args.num_heads,
                    ).to(device)
    net.eval()
    # ckpt = torch.load(args.ckptpath, map_location=device)
    ckpt = torch.load(args.ckptpath, map_location=lambda storage, loc: storage.cuda() if device.type == 'cuda' else storage)
    net.load_state_dict(ckpt)

    # data loader
    wavdir    = os.path.join(args.datadir, 'wav')
    test_list = os.path.join(args.datadir, 'sets/test_mos_list.txt')
    test_set  = MyDataset(wavdir, test_list)
    loader    = DataLoader(test_set,
                           batch_size=1,
                           shuffle=False,
                           collate_fn=test_set.collate_fn)

    l1 = nn.L1Loss()
    preds_o = {}
    preds_t = {}
    print("Start prediction...")
    with torch.no_grad():
        for wav, o_true, t_true, names in tqdm(loader):
            wav = wav.squeeze(1).to(device)
            text = get_texts_from_filename(names)
            out_o, out_t = net(wav, text)
            p_o = out_o.cpu().item()
            p_t = out_t.cpu().item()
            utt = names[0]
            preds_o[utt] = p_o
            preds_t[utt] = p_t

    # 存檔
    answer_file = os.path.join(args.expname, 'answer_track1.txt')
    with open(answer_file, 'w') as f:
        for utt in sorted(preds_o):
            f.write(f"{utt},{preds_o[utt]:.6f},{preds_t[utt]:.6f}\n")

    # 讀 ground-truth
    truth = read_file(test_list)
    pred  = {utt: (preds_o[utt], preds_t[utt]) for utt in preds_o}

    # Eval
    eval_metrics(truth, pred, "utterance level")
    eval_system_metrics(truth, pred)

if __name__ == '__main__':
    main()
