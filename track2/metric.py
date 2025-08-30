import argparse
import json

import numpy as np
import scipy.stats

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


def read_pred_csv(filepath):
    """
    Returns: dict[wavID] = [score1, score2, score3, score4]
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            # 假如第二欄是非數字，就跳過
            if not parts[1].replace('.', '', 1).isdigit():
                continue
            wavID, *scores = parts
            data[wavID] = [float(s) for s in scores]
    return data


def eval_per_score(truth_dict, pred_dict):
    keys = sorted(set(truth_dict.keys()) & set(pred_dict.keys()))
    scores_truth = np.array([truth_dict[k] for k in keys])
    scores_pred = np.array([pred_dict[k] for k in keys])

    print("\n========== UTTERANCE LEVEL ==========")
    labels = ["Production_Quality", "Production_Complexity", "Content_Enjoyment", "Content_Usefulness"]

    for i in range(4):
        t = scores_truth[:, i]
        p = scores_pred[:, i]
        mse = np.mean((t - p) ** 2)
        lcc = np.corrcoef(t, p)[0, 1]
        srcc = scipy.stats.spearmanr(t, p)[0]
        ktau = scipy.stats.kendalltau(t, p)[0]
        print(f"---- {labels[i]} ----")
        print(f"MSE  : {mse:.4f}")
        print(f"LCC  : {lcc:.4f}")
        print(f"SRCC : {srcc:.4f}")
        print(f"KTAU : {ktau:.4f}")


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


def eval_system_level(truth_dict, pred_dict):
    truth_agg = aggregate_by_system(truth_dict)
    pred_agg = aggregate_by_system(pred_dict)

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
    parser.add_argument('--truth', type=str, default='/share/nas169/wago/AudioMOS/data/track2/audiomos2025_track2_human_annotations.jsonl', required=False, help='Ground truth file (.jsonl)')
    parser.add_argument('--pred', type=str, default='/share/nas169/jethrowang/AudioMOS/track2/evaluation/l_pmr_0.5/answer.txt', required=False, help='Prediction file (.csv)')
    args = parser.parse_args()

    truth = read_truth_jsonl(args.truth)
    pred = read_pred_csv(args.pred)

    eval_per_score(truth, pred)
    eval_system_level(truth, pred)


if __name__ == '__main__':
    main()
