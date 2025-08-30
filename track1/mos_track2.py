"""
Largely adapt codes from:
https://github.com/nii-yamagishilab/mos-finetune-ssl

Modified for AudioMOS 2025 Track 2 Dataset, with minimal changes for fair comparison with Track 1.
"""
import os
import csv
import argparse
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import laion_clap
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
random.seed(1984)
import pandas as pd
import wandb
from pooling import (
    Temporal_Average_Pooling,
    Temporal_Statistics_Pooling,
    Self_Attentive_Pooling,
    Attentive_Statistics_Pooling,
    General_Self_Attentive_Pooling,
    General_Attentive_Statistics_Pooling,
    Dual_Resolution_Attentive_Pooling,
    Dual_Resolution_Statistics_Pooling,
)


class MosPredictor(nn.Module):
    def __init__(self, up_model, up_out_dim, pooling_type, segment_size):
        super(MosPredictor, self).__init__()
        self.upstream_model = up_model
        self.upstream_feat_dim = up_out_dim # 512
        self.segment_size=segment_size

        if pooling_type == 'tap':
            self.pooling = Temporal_Average_Pooling(1024)
            pool_out_dim = 1024
        elif pooling_type == 'tsp':
            self.pooling = Temporal_Statistics_Pooling(1024)
            pool_out_dim = 1024 * 2
        elif pooling_type == 'sap':
            self.pooling = Self_Attentive_Pooling(1024)
            pool_out_dim = 1024
        elif pooling_type == 'asp':
            self.pooling = Attentive_Statistics_Pooling(1024)
            pool_out_dim = 1024 * 2
        elif pooling_type == 'gsap':
            self.pooling = General_Self_Attentive_Pooling(1024, segment_size=segment_size)
            pool_out_dim = 1024
        elif pooling_type == 'gasp':
            self.pooling = General_Attentive_Statistics_Pooling(1024, segment_size=segment_size)
            pool_out_dim = 1024 * 2
        elif pooling_type == 'drap':
            self.pooling = Dual_Resolution_Attentive_Pooling(1024, segment_size=segment_size)
            pool_out_dim = 1024
        elif pooling_type == 'drsp':
            self.pooling = Dual_Resolution_Statistics_Pooling(1024, segment_size=segment_size)
            pool_out_dim = 1024 * 2
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        self.pooling_proj = nn.Linear(in_features=pool_out_dim, out_features=1024)

        # Production_Quality prediction head
        self.quality_mlp_layer1 = nn.Linear(in_features=self.upstream_feat_dim, out_features=256)
        self.quality_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.quality_mlp_layer3 = nn.Linear(in_features=64, out_features=1)

        # Production_Complexity prediction head
        self.complexity_mlp_layer1 = nn.Linear(in_features=self.upstream_feat_dim, out_features=256)
        self.complexity_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.complexity_mlp_layer3 = nn.Linear(in_features=64, out_features=1)

        # Content_Enjoyment prediction head
        self.enjoyment_mlp_layer1 = nn.Linear(in_features=self.upstream_feat_dim, out_features=256)
        self.enjoyment_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.enjoyment_mlp_layer3 = nn.Linear(in_features=64, out_features=1)

        # Content_Usefulness prediction head
        self.usefulness_mlp_layer1 = nn.Linear(in_features=self.upstream_feat_dim, out_features=256)
        self.usefulness_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.usefulness_mlp_layer3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, wavs):

        feats = self.upstream_model.get_audio_embedding_from_data(wavs, use_tensor=True).to(device)
        # feats: (B, T, D)

        wav_embed = self.pooling(feats)
        wav_embed = self.pooling_proj(wav_embed)
        wav_embed = self.upstream_model.model.audio_projection(wav_embed)
        wav_embed = F.normalize(wav_embed, dim=-1)

        # Quality prediction
        q_h1 = self.quality_mlp_layer1(wav_embed)
        q_h2 = self.quality_mlp_layer2(q_h1)
        out_quality = self.quality_mlp_layer3(q_h2)

        # Complexity prediction
        c_h1 = self.complexity_mlp_layer1(wav_embed)
        c_h2 = self.complexity_mlp_layer2(c_h1)
        out_complexity = self.complexity_mlp_layer3(c_h2)

        # Enjoyment prediction
        e_h1 = self.enjoyment_mlp_layer1(wav_embed)
        e_h2 = self.enjoyment_mlp_layer2(e_h1)
        out_enjoyment = self.enjoyment_mlp_layer3(e_h2)

        # Usefulness prediction
        u_h1 = self.usefulness_mlp_layer1(wav_embed)
        u_h2 = self.usefulness_mlp_layer2(u_h1)
        out_usefulness = self.usefulness_mlp_layer3(u_h2)

        return out_quality, out_complexity, out_enjoyment, out_usefulness

class MyDataset(Dataset):
    def __init__(self, wavdir, mos_csv_path):
        self.wavdir = wavdir
        self.df = pd.read_csv(mos_csv_path)

        self.sample_ids = self.df['sample_id'].tolist()
        self.production_quality = self.df['Production_Quality'].tolist()
        self.production_complexity = self.df['Production_Complexity'].tolist()
        self.content_enjoyment = self.df['Content_Enjoyment'].tolist()
        self.content_usefulness = self.df['Content_Usefulness'].tolist()

    def __getitem__(self, idx):

        wavname = self.sample_ids[idx]
        wavpath = os.path.join(self.wavdir, wavname)

        wav, sr = torchaudio.load(wavpath)

        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # resample
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # 截長補短：限制最長 30s (16k × 30s = 480k 樣本)
        if wav.size(1) > 480000:    # 16khz*30s
            wav = wav[:,:480000]

        return (
            wav,
            self.production_quality[idx],
            self.production_complexity[idx],
            self.content_enjoyment[idx],
            self.content_usefulness[idx],
            wavname
        )

    def __len__(self):
        # return len(self.wavnames)
        return len(self.sample_ids)

    def collate_fn(self, batch):
        wavs, quality_scores, complexity_scores, enjoyment_scores, usefulness_scores, wavnames = zip(*batch)

        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]

        padded_wavs = [torch.nn.functional.pad(w, (0, max_len - w.shape[1]), 'constant', 0) for w in wavs]
        padded_wavs = torch.stack(padded_wavs)

        return (
            padded_wavs,
            torch.tensor(quality_scores),
            torch.tensor(complexity_scores),
            torch.tensor(enjoyment_scores),
            torch.tensor(usefulness_scores),
            wavnames
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="../data/track2_data", required=False, help='Path of track2 dataset')
    parser.add_argument('--expname', type=str, required=False, default='exp_track2', help='ckpt will be saved in track2_ckpt/EXPNAME')
    parser.add_argument('--pooling_type', type=str, default='tsp', choices=['tap','tsp','sap','asp', 'gsap', 'gasp', 'drap', 'drsp'], help='Pooling type')
    parser.add_argument('--segment_size', type=int, required=False, default=1, help='Number of frames per segment in gsap, gasp, drap, drsp')
    args = parser.parse_args()

    wandb.init(
        project="AudioMOS_Track2",
        name=args.expname,
    )

    DATA_DIR = args.datadir
    UPSTREAM_MODEL = 'CLAP-music'
    EXP_NAME = args.expname
    CKPT_DIR = '../track2_ckpt/' + EXP_NAME # checkpoint will be saved here
    if not os.path.exists(CKPT_DIR):
        os.system('mkdir -p ' + CKPT_DIR)

    wavdir = os.path.join(DATA_DIR, 'wav')
    trainlist = os.path.join(DATA_DIR, 'sets/train.csv')
    validlist = os.path.join(DATA_DIR, 'sets/dev.csv')

    if UPSTREAM_MODEL == 'CLAP-music':
        UPSTREAM_OUT_DIM= 512
        model = laion_clap.CLAP_Module(enable_fusion=False,  amodel= 'HTSAT-base')
        model.load_ckpt('../upstream/music_audioset_epoch_15_esc_90.14.pt')
    else:
        print('*** ERROR *** Model type ' + UPSTREAM_MODEL + ' not supported.')
        exit()

    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=8, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    net = MosPredictor(model,
        UPSTREAM_OUT_DIM,
        pooling_type=args.pooling_type,
        segment_size=args.segment_size,
        ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9)

    PREV_VAL_LOSS = 9999999999
    orig_patience=20
    patience=orig_patience
    BEST_EPOCH = 0
    BEST_PATH = os.path.join(CKPT_DIR, 'best_ckpt')

    for epoch in range(1,1001):
        # ——Training——
        STEPS=0
        net.train()
        train_epoch_loss = 0.0
        train_epoch_pq_loss = 0.0
        train_epoch_pc_loss = 0.0
        train_epoch_ce_loss = 0.0
        train_epoch_cu_loss = 0.0


        for i, data in enumerate(tqdm(trainloader, desc="Training Progress", ncols=100), 0):
            STEPS += 1
            wavs, pq_labels, pc_labels, ce_labels, cu_labels, filenames = data

            wavs = wavs.squeeze(1)  # tensor(batch,T)
            pq_labels = pq_labels.unsqueeze(1).to(device)
            pc_labels = pc_labels.unsqueeze(1).to(device)
            ce_labels = ce_labels.unsqueeze(1).to(device)
            cu_labels = cu_labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            out_pq, out_pc, out_ce, out_cu = net(wavs)

            loss_pq = criterion(out_pq, pq_labels)
            loss_pc = criterion(out_pc, pc_labels)
            loss_ce = criterion(out_ce, ce_labels)
            loss_cu = criterion(out_cu, cu_labels)

            total_loss = (loss_pq + loss_pc + loss_ce + loss_cu) / 4.0
            total_loss.backward()
            optimizer.step()

            train_epoch_loss += total_loss.item()
            train_epoch_pq_loss += loss_pq.item()
            train_epoch_pc_loss += loss_pc.item()
            train_epoch_ce_loss += loss_ce.item()
            train_epoch_cu_loss += loss_cu.item()
        avg_train_loss = train_epoch_loss / STEPS
        print('EPOCH:' + str(epoch) + ', AVG EPOCH TRAIN LOSS: ' + str(train_epoch_loss / STEPS))
        wandb.log({'epoch': epoch,
                'avg_train_loss': avg_train_loss,
                'train_pq_loss': train_epoch_pq_loss / STEPS,
                'train_pc_loss': train_epoch_pc_loss / STEPS,
                'train_ce_loss': train_epoch_ce_loss / STEPS,
                'train_cu_loss': train_epoch_cu_loss / STEPS,
        })

        # clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        # ——Validation——
        VALSTEPS=0
        net.eval()
        valid_epoch_loss = 0.0
        valid_epoch_pq_loss = 0.0
        valid_epoch_pc_loss = 0.0
        valid_epoch_ce_loss = 0.0
        valid_epoch_cu_loss = 0.0
        for i, data in enumerate(tqdm(validloader, desc="Validating Progress", ncols=100), 0):
            VALSTEPS+=1
            wavs, pq_labels, pc_labels, ce_labels, cu_labels, filenames = data

            wavs = wavs.squeeze(1)
            pq_labels = pq_labels.unsqueeze(1).to(device)
            pc_labels = pc_labels.unsqueeze(1).to(device)
            ce_labels = ce_labels.unsqueeze(1).to(device)
            cu_labels = cu_labels.unsqueeze(1).to(device)
            with torch.no_grad():
                out_pq, out_pc, out_ce, out_cu = net(wavs)

            loss_pq = criterion(out_pq, pq_labels)
            loss_pc = criterion(out_pc, pc_labels)
            loss_ce = criterion(out_ce, ce_labels)
            loss_cu = criterion(out_cu, cu_labels)

            valid_loss = (loss_pq + loss_pc + loss_ce + loss_cu) / 4.0

            valid_epoch_pq_loss += loss_pq.item()
            valid_epoch_pc_loss += loss_pc.item()
            valid_epoch_ce_loss += loss_ce.item()
            valid_epoch_cu_loss += loss_cu.item()
            valid_epoch_loss += valid_loss.item()

        avg_val_loss=valid_epoch_loss / VALSTEPS

        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        wandb.log({'avg_val_loss': avg_val_loss,
                    'val_pq_loss': valid_epoch_pq_loss / VALSTEPS,
                    'val_pc_loss': valid_epoch_pc_loss / VALSTEPS,
                    'val_ce_loss': valid_epoch_ce_loss / VALSTEPS,
                    'val_cu_loss': valid_epoch_cu_loss / VALSTEPS,
        })

        if hasattr(net.pooling, 'alpha') and hasattr(net.pooling, 'beta'):
            wandb.log({
                'alpha': net.pooling.alpha.item(),
                'beta' : net.pooling.beta.item()
            })

        # ——Early Stopping & Save Best——
        if avg_val_loss < PREV_VAL_LOSS:    # Loss has decreased
            torch.save(net.state_dict(), BEST_PATH)
            BEST_EPOCH = epoch
            PREV_VAL_LOSS = avg_val_loss
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
    # 重新命名最佳檔案
    os.rename(BEST_PATH, os.path.join(CKPT_DIR, 'best_ckpt_'+str(BEST_EPOCH)))
    print('Finished Training, best epoch:', BEST_EPOCH)

if __name__ == '__main__':
    main()
