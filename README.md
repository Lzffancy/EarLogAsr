# EarLogAsr

长音频的本地离线分析管线。输入是几小时的连续录音，输出是带说话人、场景、置信度标记的结构化转写（JSONL），供上层 LLM 做校准、摘要和日程还原。

设计约束：中文语音、CPU 可跑（有 GPU 自动加速）、每天约 8 小时录音量、不允许丢失任何语音内容、音频不出本机。

## 为什么做这个项目

人一天说七八个小时的话，绝大部分没有第二次机会被听见：会上答应的事、电话里给的判断、随口说出的决定依据。记不住往往不是因为不重要，而是因为没留下可回放的原文。更麻烦的是记忆本身会重构——你记住的是你对自己的叙事，不是当时真正说出口的字。

想要的工具是：戴上麦克风就不用管，一天结束后能拿回"今天到底说了什么、对谁说的、在什么场合说的"。

现成的方案都对不上：

| 现成方案 | 为什么对不上 |
|---|---|
| 会议助手（Otter / 妙记类） | 以"一场会"为单位、要手动开始；素材强制上云 |
| 云转写 / 语音 API | 按分钟计费、音频出厂；只给文本，不给说话人身份和场景 |
| 本地跑 Whisper | 文本能出，但长音频的切分、去重、说话人、断点续跑全靠自己搭；低音量段召回不稳 |
| Rewind 类屏幕回忆 | 抓屏幕，抓不到线下真实发生的对话 |

**定位**：把"全天物理世界里真实发生的对话"转成结构化文本流，作为上层 LLM 复盘的原料。这一层只负责忠实还原发生了什么，不做判断、不下结论——判断属于 LLM 层，而 LLM 层只读 JSONL、从不碰音频。

上层产品形态是"语音型第二大脑"：不靠关键词检索，靠回放全天对话做自我觉察和认知迭代。ASR 管线是它的地基，也就是本仓库的全部内容。目标是**用 20 分钟的机器时间，换回一整天的原文**：一字不丢、一人不错、全部留在本机。

## 优势与特点

### 1. 第一优先级是"不丢音"，不是"降噪提纯"

这是全项目最核心的取舍。主流做法是 VAD 先切人声再送 ASR，图的是省算力、少幻觉；但只要 VAD 在 ASR 之前把一段真人语音判成静音，后端换什么模型都救不回来。实测双 VAD（fsmn + silero）取并集，低音量远场段仍然漏 60 处。

所以这里反过来：**用重叠滑窗无条件覆盖全音频，把"是不是语音"的判断挪到识别之后**——判断依据从"听起来像不像语音"变成"识别出来像不像话"。纯噪声窗由退化过滤器拦截（纯标点、单字占比 >0.7），误杀率明显更低。代价是算力约 1.5 倍，离线批处理认这笔账。

### 2. 全链路本地，没有云端依赖

不调 API、不需要 key、不联网也能跑；录音、转写、声纹全部留在本机。这不是可选项而是红线：全天候录音本质上是把一个人的私生活完整记录下来，任何"上传后删除"的承诺都不成立。

代价是自己扛模型体积和算力——也正因如此，才需要下面这些工程优化。

### 3. 三流解耦，JSONL 是唯一契约

内容流（说了什么）/ 说话人流（谁说的）/ 场景流（在什么地方）互相独立，各自落盘、各自可重跑，最后由 `fuse.py` 按时间轴对齐。好处：

- 任意一流可以单独换模型、单独调参，不影响另外两流
- embedding 全部保留为 `.npy`，调聚类参数用 `--recluster` 复核，不用重跑 ASR
- ASR 层与 LLM 层的接口只有一个 JSONL 文件，两层可以独立迭代、独立换人做

### 4. 说话人是"身份"，不是"编号"

纯聚类在远场小窗上会分裂、会并簇（同一个人户外/室内被分成两簇、两个汇报人被并成一簇）。这里用**声纹注册 + 验证**兜底：注册过的本人直接标姓名，未命中才回落聚类。实测分离度够用（本人命中段相似度 >0.40，他人 p90 = 0.19）。

结果不是一个漂移的 `SPK03`，而是"这个人是李卓凡"——后续按人聚合、按人复盘才有意义。

### 5. 速度：8 小时录音 20 分钟跑完，且瓶颈已经找对

不是靠堆硬件，是靠认清瓶颈在哪：

- GPU 只用在划算的地方：内容流 6.6×、场景流 5.5×；说话流只有 1.4× 且有崩溃风险，**默认锁 CPU**
- 真正的浪费是进程初始化：`import torch` 28.7s + `import funasr` 11.7s，旧流程每个文件付三遍，一天 16 个文件白烧 36 分钟
- `run_day.py` 一次 import、一次加载，GPU 链与 CPU 链两个进程并行 → 全流 2.4×，8h 从 58min 压到 20min

同类本地方案的常见耗时是"几个小时"，量级差别就在这里。

### 6. 有第二通道兜底，不是一次性输出

低置信段（叠字/重复率高、语速异常、标点占比异常）会被挑出来交给质量优先的 FireRedASR2 复跑一次，结果回灌替换，时间戳同时升级为句级精确值。复核预算默认封顶音频时长 20%，按严重度加权取舍——不会被少量烂段拖垮整体耗时。

### 7. 每个选型都有账本，包括"此路不通"

README 里每个技术选择都附实测数据和放弃理由，也记录**已实测否决的方向**（fp16、批量推理、改线程数、GPU 跑说话人流）。这不是文档洁癖：这类项目的最大成本是反复试错，把"为什么不用 X"写下来，下次就不会有人再试一遍。

### 边界

- 不适用于实时流式场景（重叠滑窗天然是批处理，延迟 30s 起步）
- 不解决语义场景判断：声学只能给"地铁/餐厅/安静"，是不是周会得靠文本层反推
- 8 小时 20 分钟是"全链跑完"的时间，不是"当天实时可用"
- 说话人时长统计在远场小窗下仍有偏差，turn 切换点可靠、时长偏保守

## 架构

```
                    ┌─ speaker_timeline.py  CAM++ 小滑窗声纹 → 说话人流
原始音频（全长） ────┼─ scene_timeline.py    PANNs 场景标签   → 场景流
                    └─ slidewin_pipeline.py SenseVoice 滑窗   → 内容流
                                      │
                          fuse.py 三流对齐到统一时间轴
                                      │
                fused_xxx.jsonl {start, end, spk, scene, text, conf}
                                      │
                    （可选）firered_review.py 低置信段复核回灌
                                      │
                              LLM 层（不碰音频）
```

核心原则：**ASR 层与 LLM 层的唯一接口是 JSONL**。LLM 层只读文本不碰音频，两层可独立迭代。

## 各模块构成与选型依据

### 内容流：无 VAD 重叠滑窗（`slidewin_pipeline.py`）

30 秒窗、20 秒步长（10 秒重叠）覆盖全音频，每窗独立送 SenseVoice-Small，窗口间用模糊后缀-前缀对齐（difflib，ratio≥0.70）去重，串行合并出全文。

**为什么不走 VAD 前置切分**：实测 VAD（fsmn-vad、silero-vad 单独或并集）在低音量远场段会判静音并丢弃语音。丢失发生在 ASR 之前，后端模型再强也见不到这段音频——MIC006 实测双 VAD 并集仍漏 60 处。滑窗方案把"是否是语音"的判断从声学前验改成识别后验：纯噪声窗由退化过滤器拦截（输出为纯标点、或单字符占比 >0.7 则丢弃），判断依据是"识别出来像不像话"而非"听起来像不像语音"，前者误杀率明显更低。

**代价与边界**：重叠区重复计算，算力约为 VAD 方案的 1.5 倍，离线批处理可接受；实时流式场景不适用。单窗 30 秒是 SenseVoice 的有效长度上限，更长会截断。

### 说话人流：注册验证 + 聚类混合（`speaker_timeline.py`、`enroll_speaker.py`）

CAM++ 在 4 秒窗 / 2 秒步长上逐窗提取声纹 embedding，两种模式：

- **聚类模式**（默认）：层次聚类（余弦距离）或 `EARLOG_N_SPK=k` 谱聚类，适合完全未知的录音
- **验证模式**（`--verify voiceprints.json`）：每窗与注册声纹质心算余弦相似度，≥阈值（默认 0.45）直接标姓名；未命中回落聚类

`enroll_speaker.py` 从已有转写稿按说话人标签抽段，平均 embedding 得注册质心。

**为什么混合**：纯聚类在远场小窗上会分裂或并簇（实测同一人户外/室内被分成两簇、两个汇报人并簇）；注册验证分离度实测良好（本人命中段相似度 >0.40，他人 p90=0.19），但只覆盖注册过的人。回落聚类保证未知说话人仍有标签。

**已知边界**：turn 切换点可靠；说话人时长统计在远场小窗下有偏差；聚类并簇会连带影响按簇的自动改名。

### 场景流（`scene_timeline.py`）

PANNs CNN14 在 10 秒窗上出 AudioSet 标签（白名单过滤到生活场景类别）。无人声段的场景（地铁、雨声、街道）只有声学路径能拿到——内容流在无人声时不出文本，文本层无米下锅。

**为什么不用 SenseVoice 的事件标签**：实测其在非语音段几乎不产出（48 个无人声段只报出 2 个标签），且噪声段会幻听文本。

### 融合（`fuse.py`）

三流按时间轴对齐：每条文本按时间重叠量分配说话人 turn 和场景标签，产出统一 JSONL：

```
{"start_ms", "end_ms", "spk", "scene", "text", "emotion", "conf"}
```

### 复核第二通道（`firered_review.py`，可选）

SenseVoice 是速度优先（RTF≈0.07）；FireRedASR2-AED 质量优先（句级原生时间戳、低音量召回最强，但 RTF 0.2–0.7、模型 4.7GB）。分工：SenseVoice 跑全量，FireRedASR2 只复核低置信段。

- 低置信判定：叠字/重复率 >0.30、标点占比 >0.35、活跃段语速 <1 字/秒；相邻标记合并成 ≤90s 复核段
- 复核预算默认封顶音频时长的 20%（`EARLOG_REVIEW_MAX_FRAC`），按严重度加权取舍
- FireRedVAD 阈值默认降到 0.25（`FIRERED_VAD_THRESHOLD`），否则滤掉的正是要复核的低音量语音；FR 返回空的段自动降级无 VAD 重试一次；仍空保留 SenseVoice 原文——这是 FR 的真实能力下限
- 复核结果用 `fuse.py --review` 回灌：FR 句子替换对应区间，时间戳升级为句级精确值，标记 `conf=reviewed`

### 说话人锚定（`anchor_speakers.py`）

扫描自报家门句式（"大家好我是X"）把聚类标签映射为姓名。带问候语境的判高置信才自动应用；其余保留证据供人工确认。注册表存 `speakers_registry.json`，`fuse.py --names` 应用。

## 使用

```bash
# 整日批处理（推荐）：一次 import、一次加载，GPU 链与 CPU 链并行
python run_day.py <wav_dir> <out_dir> --verify voiceprints.json

# 单文件调试：三条流分开跑
python slidewin_pipeline.py <wav> sw_xxx [--jobs N]
EARLOG_N_SPK=5 python speaker_timeline.py <wav> spk_xxx
python scene_timeline.py <wav> sc_xxx

# 融合
python fuse.py fused_xxx --text sw_xxx.json --spk spk_xxx.json --scene sc_xxx.json

# 可选：复核 + 回灌 + 锚定
python firered_review.py <wav> sw_xxx sw_xxx_review [--ranges=24:00-30:00] [--all]
python anchor_speakers.py fused_xxx.jsonl
python fuse.py fused_xxx --text sw_xxx.json --spk spk_xxx.json --scene sc_xxx.json \
    --review sw_xxx_review.json --names speakers_registry.json

# 可选：声纹注册 + 验证模式
python enroll_speaker.py 姓名 fused_xxx.jsonl SPKxx
python speaker_timeline.py --recluster spk_xxx_emb.npy spk_xxx_v --verify voiceprints.json
```

每个 `*_emb.npy` 都保留，调聚类参数用 `--recluster` 免重跑 ASR。

## 优化设计

实测环境 i7-9700（8 核 CPU），30 分钟音频：内容流 133s、说话人流 75s、场景流 44–104s，全链合约 4 分钟；8 小时录音约 70 分钟。

**GPU 实测**（2026-10-21，RTX 3050 Laptop 4GB，30 分钟真实录音同文件对照）：

| 流 | CPU | GPU | 加速 | 输出 |
|---|---|---|---|---|
| 内容流 SenseVoice | 177s | 27s | **6.6×** | 56 段 / 651 字逐字一致 |
| 场景流 PANNs | 33s | 6s | 5.5× | 标签完全一致 |
| 说话人流 CAM++ | 78s | 54s | 1.4× | emb 差 1.2e-3，聚类一致 |

单窗 forward 2.0s → 0.23s，端到端 rtf 0.098 → 0.015。

**但纯推理不是瓶颈，进程初始化才是**（2026-10-23 实测）。单文件固定开销 **49.9s**：

| 项 | 耗时 |
|---|---|
| `import torch` + torchaudio | **28.7s** |
| `import funasr` | **11.7s** |
| 加载 SenseVoice | 7.4s |
| 读 259MB wav + resample | 2.1s |

旧流程每个文件起 3 个进程，这笔钱付 3 遍。16 文件的一天 ≈ 36 分钟纯浪费在
重复 import 上。`run_day.py` 把它压成 1 次，并让 GPU 链（SenseVoice + PANNs）
与 CPU 链（CAM++）并行跑，实测：

- 3 个 30min 文件、仅 asr：207s → **96.6s（2.14×）**
- 3 个 30min 文件、全流：约 654s → **276s（2.4×）**
- 8 小时外推：约 58min → **约 20min**

**已知代价**：同一进程里第 2 个及以后的文件，转写结果与"冷启动单跑同一文件"
有约 0.7% 的字符差异（48/55 段完全一致）。这是模型冷/热态差异，**稳定而非随机**
——三次 run_day 运行之间 55/55 完全一致。`torch.use_deterministic_algorithms`
和 `CUBLAS_WORKSPACE_CONFIG` 都不能消除它。要严格对齐旧产物就用 `run_batch.py`。

- **RMS 能量门**：真静音窗（RMS<1e-3）直接跳过 ASR。与 VAD 不同——只拦"什么声音都没有"，不做语音/噪声判断，零误杀风险。对含大段空白的录音（如夜间）节省线性时间
- **`--jobs N` 进程池**：窗口级并行。实测本机无增益（单进程 torch 已占满 8 核），保留给多核大机。注意不同线程数改变浮点归约顺序，输出有 ±0.5% 字面漂移
- **已验证无效的方向**（2026-10-23 全部实测）：
  - SenseVoice 批量推理只快 6%；CAM++ 批量 bs=8 只快 5.7%、bs≥16 反而更慢（但输出逐位一致）
    → 瓶颈是 CPU 卷积算力，不是 Python 调用开销
  - `fp16=True` 直接报错：FunASR `auto_model.py:678` 只转权重不转输入，权重 half 输入 float32
  - `torch.autocast` 只快 1.07×，代价是逐窗文本一致率掉到 62%、整段相似度 0.9647 → **维持 fp32**
  - 改 `torch.set_num_threads` 会改变浮点归约顺序，4 线程 vs 8 线程只有 48/55 段一致 → 不要动
  - CPU 饱和机器上进程并发为负收益
- **未做的杠杆**：int8 量化（约 2 倍）、重叠减到 5s（省 17%）、
  把 CAM++ 的 hop 从 2s 放到 4s（spk 算力减半，但 turn 时间分辨率从 2s 降到 4s，需验收）

## 兼容性

`device_config.py` 自动探测设备：`torch.cuda.is_available()` 为真走 `cuda:0`，否则 CPU。`EARLOG_DEVICE=cuda:0/cpu` 强制覆盖，无需改代码。

**当前实际精度是 fp32，不是 fp16**：FunASR 的 `AutoModel` 没有传 `use_half`，`device_config.py` 也只返回设备名，代码里没有任何 `.half()`。所以 GPU 上跑的是 fp32，fp16 仍是未兑现的加速余量。

`speaker_device()` 是唯一例外：**说话人流默认锁定 CPU**，即使检测到 GPU。实测 CAM++ 在 GPU 上只有 1.4× 收益，却有约 60% 概率进程级静默崩溃（无 traceback、不落盘，只在首次崩时吐 `TORCH_USE_CUDA_DSA` 提示）。要赌就用 `EARLOG_SPK_DEVICE=cuda:0`，并自行加重试。

GPU 模式注意：

- `slidewin_pipeline.py` 的 `jobs` 会被强制为 1（多进程抢一张卡只会 OOM；GPU 本身够快）
- 装 CUDA 版 torch：`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`
- 3050 4G 笔记本实测显存峰值 **1872 MiB**（PANNs + CAM++ 会话），4GB 够用但余量不到一半，别同时开别的吃显存程序
- **CUDA 版 torch 即使 `EARLOG_DEVICE=cpu`，import 时照样加载 cuDNN DLL**。Windows 页面文件偏小时会报 `OSError: [WinError 1455] 页面文件太小`，表现为启动阶段偶发失败（重跑通常就过），不是代码 bug，调大虚拟内存即可

其它：

- Python 3.10 验证通过；**torch 与 torchaudio 版本必须一致**（不一致在 Windows 报 WinError 127）
- panns_inference 官方下载脚本依赖 wget，Windows 需手动下载权重到 `~/panns_data/`
- silero-vad（历史模块用）torch 模型带循环状态，必须逐 chunk 顺序推理，批量输入输出失真
- FireRedASR2 不内置于本仓库：`FIRERED_REPO` / `FIRERED_AED_DIR` / `FIRERED_VAD_DIR` 指向外部路径

## 模块

- `run_day.py`：整日批处理入口。一次 import、一次加载，GPU 链与 CPU 链并行
- `run_batch.py`：逐文件起子进程的旧批处理入口。慢 2.4×，但产物与"单文件冷启动"逐字一致，留作对照与调试
- `sensevoice_transcribe.py`：SenseVoice 富标签解析（parse_rich / fmt_ts），也可独立转写
- `earlog_pipeline.py`：聚类与段合并工具库（cluster / merge_segments）
- `device_config.py`：设备自动探测，被所有管线脚本 import，勿删
- `firered_review.py`：复核第二通道
- `enroll_speaker.py` / `anchor_speakers.py`：声纹注册 / 自报家门锚定

## 方案取舍记录

| 取舍 | 选择 | 放弃 | 依据 |
|---|---|---|---|
| 切分 | 无 VAD 滑窗 | VAD 前置 | 双 VAD 并集仍漏 60 处低音量语音（MIC006 实测） |
| 主力 ASR | SenseVoice-Small | Paraformer / Qwen3-ASR / Fun-ASR-Nano | 速度 ×2、低音量召回最强、CPU 可行；Paraformer 仅时间戳占优 |
| 复核 | FireRedASR2-AED 按需复核 | 全量跑 FR | RTF 0.2–0.7，8 小时录音 CPU 跑不动全量 |
| 说话人 | 注册验证+聚类混合 | 纯聚类 | 远场小窗聚类分裂/并簇 |
| 场景 | PANNs 专用 AED | SenseVoice 事件标签 | 后者在非语音段几乎不产出 |
| 情绪 | SenseVoice 顺带输出 | 专用模型 | 嘈杂段误报多，只当辅助信号 |
| 并发 | 进程池（保留） | 线程级并发 | torch 内部已多线程，线程并发互相抢核 |
