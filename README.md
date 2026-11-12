# 材料文献带隙信息抽取 v2

本系统从给定论文段落抽取目标材料的显式带隙。模型选择原文能量位置并给出观测来源，程序还原数值和单位，计算 eV，保留证据原句及字符区间。最终抽取 JSON 与第一阶段的四字段任务 Schema 兼容。

## 包内结构

| 文件 | 用途 |
| --- | --- |
| `task_schema.json` | 输入、内部标签协议、外部结果、单位与校验契约 |
| `energy.py` | 精确跨度、量纲、范围、区间与非目标物理量的可审计规则 |
| `knowledge.json` | 领域术语表、材料到观测的本体关系、八张检索卡和独立 few-shot 示例 |
| `engine.py` 与 `verify.py` | 提示编译、模型调用、证据复核、来源校正和一一匹配 |
| `run.py` | 提示生成与优化、对照实验、重跑、盲评导出、交互抽取 |
| `build_data.py` | 从 PV600 的专家标注及偏移生成冻结数据 |
| `test_engine.py` | 单元测试 |
| `data/` | 冻结的正式样本、排除清单、受控样本和演练参考 |
| `results/` 与 `confirmation/` | 诊断、独立确认、原始调用、候选轨迹、盲评和演练审计 |

## 运行环境

Python 3.10 或以上。程序只用标准库。推理使用 llama.cpp b6500 与 Qwen3-4B-Instruct-2507 Q4_K_M。第一次下载需要联网和约 2.6 GB 磁盘空间，随后可离线运行。实验在 Windows x64、NVIDIA RTX 3050 Laptop GPU（Vulkan）运行。系统 ZIP 不重复附带模型权重和运行时。

在解压后的 `BandGapPrompt` 文件夹开启 PowerShell。

```powershell
New-Item -ItemType Directory -Force runtime | Out-Null
curl.exe -fL --retry 3 "https://github.com/ggml-org/llama.cpp/releases/download/b6500/llama-b6500-bin-win-vulkan-x64.zip" -o runtime/llama.zip
Expand-Archive -LiteralPath runtime/llama.zip -DestinationPath runtime/llama -Force
curl.exe -fL --retry 3 "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" -o runtime/qwen.gguf
(Get-FileHash runtime/qwen.gguf -Algorithm SHA256).Hash
```

校验模型 SHA-256 为 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`。如使用 CPU，请将下载 URL 的 `win-vulkan-x64` 改为 `win-cpu-x64`，把以下启动命令中的 `-ngl 99` 改为 `-ngl 0`。其他平台从同一发行标签下载对应引擎，保留参数。CPU 运行时间较长。

## 快速验证与交互运行

第一个终端以两个并发槽启动评估和演示服务。

```powershell
.\runtime\llama\llama-server.exe -m runtime/qwen.gguf -a Qwen3-4B-Instruct-2507-Q4_K_M -c 6144 -np 2 -t 8 -ngl 99 --port 8279
```

在第二个终端进入同一文件夹。

```powershell
$env:PYTHONIOENCODING = "utf-8"
curl.exe --noproxy 127.0.0.1 http://127.0.0.1:8279/health
python -m unittest -v
python run.py extract --input data/demo_real.json --reference data/demo_reference.jsonl --audit demo_audit.json
python run.py live --audit live_audit.json
```

`live` 从键盘读取目标化学式、可选文献标识、若干句子和结束空行。现场由听众给出未加入本系统优化或测试的段落。若终端显示引擎错误，保留错误信息并按 README 检查模型版本、可用内存和监听端口。服务运行完可在第一个终端按 Ctrl+C 退出。

系统将能量提及编号为 `g1` 等。模型内部返回如 `{"g1":"exp"}`，公开观测返回如 `[[3420,"meV","exp",1]]`。完整档案含 `band_gap_eV`、`evidence` 和从 0 起算的原句半开区间 `span=[start,end)`。无匹配观测时输出 `[]`。保留原始响应、锚点阶段结果和每条复核操作。复核器拒绝可确认的异材料或非带隙标签，用局部文献、实验和计算线索核对来源，无明确归属时写 unk。没有可选能量提及时确定性弃权。复核规则存在保守误拒与漏检，报告给出实测案例。若提供 `--reference`，系统在推理完成后才用同源且完全相同的段落指纹查找独立参考，返回匹配、冲突或未收录状态。

## 完整复现正式实验

1. 第一个终端用 `-np 4` 启动开发服务，保持 `-c 6144` 和其他参数一致。第二个终端生成并冻结三轮候选轨迹。

```powershell
python run.py bench --design-only --workers 4 --out reproduced
```

2. 停止开发服务，以快速验证一节的 `-np 2` 重新启动。单槽容量从 1536 变为 3072 token，能容纳正式样本中符号密集的较长段落。五组正式测试都用相同设置。运行以下命令。

```powershell
python run.py bench --resume-design --workers 2 --out reproduced
python run.py confirm --design reproduced --out reproduced_confirmation
```

第一条命令验证冻结策略、开发数据和提示指纹，运行 40 条诊断、受控样本和复跑。候选反思使用锚点阶段评分，五组最终结果统一经证据复核。第二条命令在 30 条此前未用过的文章上执行独立确认。若只复跑诊断，复制 `results/policies.json` 与 `results/optimization.json` 到新目录，然后用 `--resume-design --out 新目录`。参数及硬件差异可能改变个别模型响应。要用已归档的原始调用复核而不重复推理，可运行 `python run.py reprocess --from-results results --out reverified`。

3. 如需从发布源重建数据，执行下列命令。源文件 SHA-256 为 `e187209dd9de14ca10b9c18da9461f4b065bdfe61f342084877f058a8d8e3606`。

```powershell
curl.exe -fL --retry 3 "https://zenodo.org/records/15124019/files/PV600.csv?download=1" -o runtime/PV600.csv
python build_data.py runtime/PV600.csv --out rebuilt/pv600_subset.jsonl --confirm-out rebuilt/pv600_confirm.jsonl
Get-FileHash data/pv600_subset.jsonl,rebuilt/pv600_subset.jsonl -Algorithm SHA256
Get-FileHash data/pv600_confirm.jsonl,rebuilt/pv600_confirm.jsonl -Algorithm SHA256
```

诊断集含 40 篇文章的 40 个样本，20 个有目标标量，20 个没有。开发集 15 条取自已暴露文献。诊断来源排除上一版三轮用过的 135 篇文章。复核规则用开发例调试，在 40 条上复查已归档响应，并在独立 30 条重新调用。确认集 12 正 18 负，与此前所有文章标识隔离。抽样规则、公开标注偏移和原文随包提供。六条人工设计的压力样本另行统计。

## 指标与人工盲评

严格正确要求全部金标记录一一匹配且无多报。原单位、来源和证据句需匹配，归一化数值容差 0.01 eV。格式按内部 JSON 协议核验，支持唯一完整 JSON 代码块。在线成本按模型返回的输入与输出 token 计，优化调用和候选试跑另计离线成本，并按 40 条摊销。表格另列微平均记录 F1、忽略方法的证据匹配 F1、只比较数值的 F1 与配对 bootstrap 区间。

独立盲评优先选择同一批 10 条确认样本（5 正例、5 负例），覆盖三组主对照，共 30 份匿名输出。完整 CSV 也包含两项消融。评分工作簿的「盲评录入」可另存为 CSV UTF-8。保留原始列名，并填写 1–5 的整数、评审代号与评审身份。保留组织者用的 `confirmation/expert_key.json`，盲评时只发送评分表。专家完成后执行以下命令，再按各组 `priority_mean_1_5`、`priority_n` 更新报告。

```powershell
python run.py expert-report --ratings 评审A.csv 评审B.csv --out reviewed/expert_summary.json
```

每条输出先汇总评审间均值，再按组汇总。脚本拒绝重复的评审与输出组合。只有实际填报才会产生分数。空白表仅表示待评。

## 署名和模型使用

数据来自 Sipilä 等的 [PV600 数据集](https://doi.org/10.5281/zenodo.15124019)（CC BY 4.0）及[数据论文](https://doi.org/10.1038/s41597-025-05637-x)。固定人工基线采用其[问答研究](https://doi.org/10.1038/s43246-025-00979-w)中公开的问题，仅替换材料占位符，所有实验组使用同一契约包装器。演练参考值来自 [WSe₂ 光谱研究](https://doi.org/10.1038/s41467-019-11751-3)，录入短句按论文事实转述。

Qwen 用于直接生成基线、生成初始流程、开发反思与改写以及抽取。Python 负责锚点检测、单位换算、检索、校验和评分。Codex 协助实现代码、整理报告与失败分析。论文来源与模型角色在总包文档列明。
