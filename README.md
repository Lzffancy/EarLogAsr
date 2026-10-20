# EarLogAsr

录音笔长音频的本地离线分析管线（中文场景，CPU 可跑，有 GPU 自动加速）。

架构：ASR 层三条流 → 统一时间轴 JSONL → LLM 层还原。

## 设备兼容

所有脚本通过 `device_config.py` 自动探测设备：`torch.cuda.is_available()` 为真就走 `cuda:0`，否则回落 CPU。无需改代码。

```bash
# 强制指定设备（可选）
EARLOG_DEVICE=cpu python slidewin_pipeline.py ...     # 有 GPU 也强制用 CPU
EARLOG_DEVICE=cuda:0 python slidewin_pipeline.py ...  # 显式指定卡
```

GPU 模式下注意：

- `slidewin_pipeline.py` 的多进程参数 `jobs` 会被强制为 1（多进程抢一张卡只会 OOM）；GPU 本身够快，不需要多进程
- GPU 机器装 CUDA 版 torch：`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`（torch 与 torchaudio 版本必须一致）
- 3050 4G 笔记本实测可跑全部三条流（SenseVoice/CAM++/PANNs 都是小模型，显存占用 <1GB）

## 管线

```bash
# 1. 内容流：无 VAD 滑窗（30s窗/20s步长）+ SenseVoice + 重叠对齐去重
python slidewin_pipeline.py <wav> sw_xxx

# 2. 说话人流：CAM++ 小滑窗声纹聚类（EARLOG_N_SPK=k 走谱聚类）
python speaker_timeline.py <wav> spk_xxx
python speaker_timeline.py --recluster spk_xxx_emb.npy spk_xxx [阈值]  # 免重跑调聚类

# 3. 场景流：PANNs CNN14 场景标签
python scene_timeline.py <wav> sc_xxx

# 4. 融合：三流对齐到统一时间轴
python fuse.py fused_xxx --text sw_xxx.json --spk spk_xxx.json --scene sc_xxx.json

# 5. 复核（可选）：低置信段用 FireRedASR2-AED 重转写，拿回完整句+精确时间戳
python firered_review.py <wav> sw_xxx sw_xxx_review
python firered_review.py <wav> sw_xxx sw_xxx_review --ranges=24:00-30:00  # 手动指定段
```

复核通道说明：

- 低置信判定：叠字/重复率 >0.30、标点占比 >0.35、活跃段语速 <1 字/秒；相邻标记合并成 ≤90s 复核段
- 复核预算默认封顶音频时长的 20%（`EARLOG_REVIEW_MAX_FRAC` 可调），按严重度排序取舍
- 依赖仓库外的 FireRedASR2S（不入库）：`FIRERED_REPO` / `FIRERED_AED_DIR` / `FIRERED_VAD_DIR` 指路径
- FireRedVAD 阈值默认降到 0.25（`FIRERED_VAD_THRESHOLD`），否则会滤掉正是要复核的低音量语音；`FIRERED_REVIEW_VAD=0` 可整体关 VAD（召回略高但没有段内时间戳）
- AED 对极低音量段会返回 `<sil>` 被判空——这是真实能力下限，此时报告里以 SV 原文为准

## 模块

- `sensevoice_transcribe.py`：SenseVoice 富标签解析（parse_rich / fmt_ts），也可独立转写
- `earlog_pipeline.py`：聚类与段合并工具库（cluster / merge_segments）
- `device_config.py`：设备自动探测（GPU/CPU 回落），被所有管线脚本 import，勿删
- `firered_review.py`：复核第二通道（低置信段 → FireRedASR2-AED 重转写）

## 依赖

- funasr 1.4.x（SenseVoiceSmall / fsmn-vad / cam++，模型走 ModelScope 缓存）
- torch + torchaudio（版本必须一致）
- silero-vad、panns_inference（PANNs 权重需手动放到 ~/panns_data/）
- soundfile、numpy、scikit-learn

输出 JSONL 每行：`{start_ms, end_ms, spk, scene, text, emotion, conf}`，供 LLM 层消费。
