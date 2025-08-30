"""
Largely adapt codes from:
https://github.com/nii-yamagishilab/mos-finetune-ssl
"""
import os
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
from utils import *
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

        # 選擇 pooling 類型並初始化對應投影層
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
        # 統一的 projection 層，將 pool_out_dim 投影回 upstream_feat_dim
        self.pooling_proj = nn.Linear(in_features=pool_out_dim, out_features=1024)

        # 整體印象分支 MLP
        self.overall_mlp_layer1 = nn.Linear(in_features = self.upstream_feat_dim, out_features = 256)
        self.overall_mlp_layer2 = nn.Linear(in_features = 256, out_features = 64)
        self.overall_mlp_layer3 = nn.Linear(in_features = 64, out_features = 1)
        # 對齊度分支 MLP（輸入為 音頻 + 文字 嵌入串接）
        self.textual_mlp_layer1 = nn.Linear(in_features = self.upstream_feat_dim*2, out_features = 256)
        self.textual_mlp_layer2 = nn.Linear(in_features = 256, out_features = 64)
        self.textual_mlp_layer3 = nn.Linear(in_features = 64, out_features = 1)

    def forward(self, wavs, texts):
        feats = self.upstream_model.get_audio_embedding_from_data(wavs, use_tensor=True).to(device)
        # feats: (B, T, D)

        wav_embed = self.pooling(feats)     # → (B, D) or (B, 2*D)
        wav_embed = self.pooling_proj(wav_embed)  # → (B, D)

        # 然後你要送到原投影層，再 normalize：
        wav_embed = self.upstream_model.model.audio_projection(wav_embed)   # → (B, 512)
        wav_embed = F.normalize(wav_embed, dim=-1)

        text_embed = self.upstream_model.get_text_embedding(texts,  use_tensor = True).to(device)
        # 串接做對齊度分支
        combine_embed=torch.cat((wav_embed,text_embed),dim=1) # bs*1024        
        # 整體印象預測
        hidden1 = self.overall_mlp_layer1(wav_embed)
        hidden1_2 = self.overall_mlp_layer2(hidden1)
        out1 = self.overall_mlp_layer3(hidden1_2)

        # 對齊度預測
        hidden2 = self.textual_mlp_layer1(combine_embed)
        hidden2_2 = self.textual_mlp_layer2(hidden2)
        out2 = self.textual_mlp_layer3(hidden2_2)
        return out1, out2
    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        # 讀入 <檔名, 整體分, 對齊分> 的 mapping
        self.mos_overall_lookup = { }
        self.mos_coherence_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]  # 'audiomos2025-track1-S002_P044.wav'
            mos_overall = float(parts[1])
            mos_coherence = float(parts[2])
            self.mos_overall_lookup[wavname] = mos_overall
            self.mos_coherence_lookup[wavname] = mos_coherence

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_overall_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        # 截長補短：限制最長 30s (16k × 30s = 480k 樣本)
        if wav.size(1) > 480000:    # 16khz*30s
            wav = wav[:,:480000]
        overall_score = self.mos_overall_lookup[wavname]
        coherence_score = self.mos_coherence_lookup[wavname]
        return wav, overall_score, coherence_score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):
        # Batch pad：對齊到 batch 裡最長訊號長度
        wavs, overall_scores, coherence_scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        overall_scores  = torch.stack([torch.tensor(x) for x in list(overall_scores)], dim=0)
        coherence_scores  = torch.stack([torch.tensor(x) for x in list(coherence_scores)], dim=0)
        
        return output_wavs, overall_scores, coherence_scores, wavnames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="../data/MusicEval-phase1", required=False, help='Path of musiceval dataset')
    parser.add_argument('--expname', type=str, required=False, default='exp', help='ckpt will be saved in track1_ckpt/EXPNAME')
    parser.add_argument('--pooling_type', type=str, default='tsp', choices=['tap','tsp','sap','asp','gsap', 'gasp', 'drap', 'drsp'], help='Pooling type')
    parser.add_argument('--segment_size', type=int, required=False, default=1, help='Number of frames per segment in gsap, gasp, drap, drsp')
    args = parser.parse_args()

    # === 在這裡加入 INFO 印出 ===
    print("==========================================================")
    print(f"INFO: Starting experiment: {args.expname}")
    print(f"INFO: Using pooling type: {args.pooling_type}")
    if args.pooling_type in ['gsap', 'gasp', 'drap', 'drsp']:
        print(f"INFO: Segment size: {args.segment_size}")
    print("==========================================================")

    # args.expname 用作 run 的名字
    wandb.init(
        project="AudioMOS",
        name=args.expname,
    )

    DATA_DIR = args.datadir
    UPSTREAM_MODEL = 'CLAP-music'
    EXP_NAME = args.expname
    CKPT_DIR = '../track1_ckpt/' + EXP_NAME # checkpoint will be saved here
    if not os.path.exists(CKPT_DIR):
        os.system('mkdir -p ' + CKPT_DIR)    

    wavdir = os.path.join(DATA_DIR, 'wav')
    trainlist = os.path.join(DATA_DIR, 'sets/train_mos_list.txt')
    validlist = os.path.join(DATA_DIR, 'sets/dev_mos_list.txt')

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
        train_epoch_loss1 = 0.0
        train_epoch_loss2 = 0.0
        for i, data in enumerate(tqdm(trainloader, desc="Training Progress", ncols=100), 0):
            STEPS += 1
            wavs, labels1, labels2, filenames = data  
            wavs = wavs.squeeze(1)  # tensor(batch,T)
            texts=get_texts_from_filename(filenames)    # list
        
            labels1 = labels1.unsqueeze(1).to(device)
            labels2 = labels2.unsqueeze(1).to(device)

            optimizer.zero_grad()
            output1,output2 = net(wavs,texts)
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            train_loss = (loss1+loss2) / 2
            loss1.backward(retain_graph=True)
            loss2.backward()

            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_loss1 += loss1.item()
            train_epoch_loss2 += loss2.item()
        avg_train_loss = train_epoch_loss / STEPS
        print('EPOCH:' + str(epoch) + ', AVG EPOCH TRAIN LOSS: ' + str(train_epoch_loss / STEPS))
        wandb.log({'epoch': epoch, 'avg_train_loss': avg_train_loss, 'train_loss1': train_epoch_loss1 / STEPS, 'train_loss2': train_epoch_loss2 / STEPS})
        
        # clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        # ——Validation——
        VALSTEPS=0
        net.eval()
        valid_epoch_loss = 0.0
        valid_epoch_loss1 = 0.0
        valid_epoch_loss2 = 0.0
        for i, data in enumerate(tqdm(validloader, desc="Validating Progress", ncols=100), 0):
            VALSTEPS+=1
            wavs, labels1, labels2, filenames = data
            wavs = wavs.squeeze(1)
            texts=get_texts_from_filename(filenames)

            labels1 = labels1.unsqueeze(1).to(device)
            labels2 = labels2.unsqueeze(1).to(device)
            with torch.no_grad():
                output1,output2 = net(wavs, texts)

            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            valid_loss = (loss1+loss2) / 2
            
            valid_epoch_loss1 += loss1.item()
            valid_epoch_loss2 += loss2.item()
            valid_epoch_loss += valid_loss.item()
        avg_val_loss=valid_epoch_loss / VALSTEPS    

        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        wandb.log({'avg_val_loss': avg_val_loss, 'val_loss1': valid_epoch_loss1 / VALSTEPS, 'val_loss2': valid_epoch_loss2 / VALSTEPS})

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
