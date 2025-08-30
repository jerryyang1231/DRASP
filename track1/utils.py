import pandas as pd

SPECIAL_S_IDS = {"S001", "S006", "S013", "S017", "S018", "S033"}    # DEMO system use special prompts list

def get_texts_from_filename(filenames):
    prompt_ids = []
    system_ids = []
    for fn in filenames:     # audiomos2025-track1-S002_P044.wav
        fn = fn.replace("audiomos2025-track1-","")
        s_id = fn.split("_")[0]                   # 分割後第一部分 → 系統 ID，如 'S002'
        p_id = fn.split("_")[1].split(".")[0]     # 第二部分去掉 .wav → prompt ID，如 'P044'
        system_ids.append(s_id)
        prompt_ids.append(p_id)

    df = pd.read_csv('../data/MusicEval-phase1/prompt_info.txt', sep='	')
    demo_df = pd.read_csv('../data/MusicEval-phase1/demo_prompt_info.txt', sep='	')
    texts = []
    for s_id, p_id in zip(system_ids, prompt_ids):
        if s_id in SPECIAL_S_IDS:   # demo_prompt_info
            # 若此系統屬於 Demo 範圍，就到 demo_df 找完整檔名對應的文字
            demo_id = 'audiomos2025-track1-' + s_id + '_' + p_id + '.wav'
            text = demo_df.loc[demo_df['id'] == demo_id, 'text'].values
        else:   # prompt_info
            # 否則到完整 prompt_info 用 Prompt ID 查
            text = df.loc[df['id'] == p_id, 'text'].values
        
        if len(text) > 0:
            texts.append(text[0])
        else:
            texts.append(None)
    return texts

