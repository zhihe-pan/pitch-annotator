# 声学特征准确性审计

日期：2026-09-12。基线：`039e1c5`；本地修复分支：`codex/acoustic-accuracy-audit`。

## 结论

**未通过“全部声学特征正确”的整体验收。** 已发现并修复明确程序错误；基础音高、统计和部分声学指标取得限定条件下的验证证据。Jitter/Shimmer/HNR 的多段拼接、脉冲倍周期问题，以及共振峰参数和筛选问题尚未解决，不应把全部导出直接当作已验证的研究结果。

本轮是准确性审计，不是完整软件认证。没有给估计器设置缺乏依据的科研误差阈值。真实材料没有独立标注真值，只能用于流程比较。

## 环境与范围

- Python 3.10.1；NumPy 2.2.6；Parselmouth 0.4.7（内嵌 Praat 6.1.38）；独立 Praat 6.4.61；PySide6 6.11.1。
- 当前环境与 README 建议的 Python 3.11/3.12 不同，本轮没有覆盖这两个 Python 版本及 Windows/Linux。
- 36 个固定种子/确定性合成样本，分别走 internal 与 external 路径；另使用 `stim/` 中两个本地音频文件作为无真值案例。至少一个为命名可识别的非言语样本，`test_audio.wav` 的来源/类别未验证。**不能称为已完成24条分层真实语料验证。**
- 稳定音高包含100/200/400 Hz ×16/44.1/48 kHz；另含升降调、八度跳变、首尾静音、间断、幅度缩放、双声道、纯噪声、20 ms短音、源滤波元音及扰动激励。
- 独立参考不调用生产计算函数产生期望值：Praat脚本核对调用/集成；NumPy FFT重叠bin积分核对频带比与COG；显式OLS核对谱斜率；数组公式核对F0汇总。
- 同算法参考一致不等于生理真值。合成元音的滤波器极点和激励周期/振幅是生成参数；经过滤波后，不能无条件等同于Praat逐脉冲测量真值。

## 已修复且有失败→通过证据的问题

| 问题 | 修复前证据 | 当前处理 |
|---|---|---|
| 高频能量比计算错误 | 100 Hz合成音完整波形HF500旧值1.000005699，独立积分5.69923e-6；HF1000旧值1.0000006997，独立积分6.99696e-7 | `Get band energy` 的高频上限由0改为明确Nyquist；频带不在Nyquist内时高频能量为0；基础与覆盖路径均修正 |
| 跨无声段虚构F0变化 | `[20,20,NaN,30,30]`半音序列返回rise=1/3，没有实际相邻有声上升 | 只比较相邻且两端有限的帧；保留既有“无可比较帧对返回0”的约定 |
| 有声时长超出原音频 | 1秒音频、含末端1.00秒帧时返回1.005秒 | 输入时间检查，帧区间终点裁到音频时长 |
| 局部补点丢失绝对时间 | 片段起点0.7秒，fallback输出首帧约0.02秒 | 低通后Sound保留原xmin；测试强制走fallback |
| CSV非法数据静默接受 | NaN/负/重复/乱序/不规则时间、Inf音高、无效标签/参数未拒绝 | 校验时间单调、均匀步长、长度与边界；正F0与非有声标签冲突时以标签为准；明确拒绝无效数据 |
| 同名音频任意匹配 | 多个候选默认取第一个 | 唯一候选正常匹配；不能消歧时要求CSV写明确音频路径 |
| 当前文件CSV重导入失效 | 导出后再编辑，再导入导出文件，旧编辑仍存在 | 导入入口保存一次状态，切换时不再把旧状态写回新导入项 |
| 短音频基础导出字段缺失 | 正常40个基础字段，短音频仅2个 | 保持完整字段集合，不可计算项返回NaN |

HF旧值不是“微小数值误差”，此前相关导出应使用修复版重新计算。审计没有修改任何历史研究CSV。

## 数值证据与逐组结论

| 指标组 | 观察 | 验收状态 |
|---|---|---|
| 稳定F0估计 | 9个频率/采样率组合，两个路径中每例的中位绝对误差最大0.06583 cents，无超过20%的gross error | 仅在已测稳定合成音上取得支持 |
| 动态F0 | 八度跳变案例出现2/195帧gross error；噪声/无声分类另见逐案例记录 | 受限；不能保证所有动态发声准确 |
| F0汇总、帧标签与时长 | 独立公式、NaN相邻帧、音频边界、导入/编辑回归通过 | 已验证测试覆盖的规则；比例按帧统计，时长是帧支持区间，不是精确声门发声起止 |
| HF500/HF1000 | 修复后与独立FFT按bin重叠宽度积分一致；独立Praat显式端点也一致 | 单段波形公式已验证；多段拼接频谱仍受接缝影响 |
| COG、谱斜率 | 六种对照，独立FFT COG最大差约1.29e-6 Hz；显式OLS斜率最大差约5.94e-14 | 实现一致；谱斜率仍是全频带dB/Hz定义，不能等同任意文献中的spectral tilt |
| 强度统计 | 同版本独立脚本的正dB帧算术均值一致 | 限定口径；不是Praat默认能量均值，未校准录音不能直接解释为真实声压级 |
| Jitter/Shimmer | 单段与同参数同版本Praat一致，但多段拼接产生额外扰动；扰动激励有倍周期识别 | 未通过一般用途准确性验收 |
| HNR | 同版本脚本一致；内嵌6.1.38与CLI6.4.61纯净谐波差最大6.5313 dB，带噪案例约0.00967 dB | 版本敏感，另有多段拼接风险；必须记录实际声学引擎版本 |
| F1/F2/F3与带宽 | 三极点元音生成参数500/1500/2500 Hz，当前导出约521.56/1493.77/1631.61 Hz；BW3约973.15 Hz，生成极点带宽150 Hz | 未通过本组合成材料的恢复检查；不能据此宣称所有真实元音错误，也不能认为算法调用成功即可用 |
| loudnessPeaksPerSec、VoicedSegmentsPerSec、F0_octave_jump_count等自定义量 | 代码口径已审查；缺少针对每种复杂发声的独立真值验证 | 未完成科学有效性验证 |

### 仍需决定或修复的重点

1. **多个片段的测量方式**：两个稳定片段的Jitter分别约4.23e-8、4.22e-7，拼接约3.64e-4；Shimmer分别约1.12e-6、3.78e-6，拼接约1.01e-4。不能直接用片段指标平均值替代，分子/分母与可接受周期筛选必须定义。已向用户提出“先限制为单段连续有声”的方案，未擅自实施口径变更。
2. **脉冲与修正F0不联动**：在150 Hz交替扰动激励样本中，floor50的PointProcess选约75 Hz周期（149脉冲），floor100约150 Hz（298脉冲）。Shimmer从约4.04e-6变为0.201639，激励振幅变化指标为0.2。证明独立脉冲提取高度依赖参数，修正F0轨迹不保证修正音质测量。
3. **共振峰设置和显示/导出**：同波形加`gender2`文件名使上限从5000变5500，测试F3变化约176–187 Hz。显示使用原波形和pitch时刻；导出使用活动段拼接、10 ms网格和带宽筛选。两个本地音频的均值差最大约32.29 Hz。建议下一步显式配置并统一两条路径，先定义筛选规则。
4. **legacy自动提取**：无人工轨迹的基础路径使用raw AC固定10 ms；GUI使用filtered AC。当前正常GUI导出提供轨迹覆盖，不能把两条API视为等价。
5. **启动与独立应用**：GUI自动检查需要预加载librosa惰性依赖，冷启动在当前环境曾超时；未修复或证明该性能问题已消失。构建成功不等于独立应用已完成全部导出操作验收。

## 操作链路和回归

- 28项 unittest（每项可包含多个数值断言）：导入边界、F0公式、HF独立积分、缺失值、时长、强制fallback时间、短音schema、参数、CSV往返、撤销等。
- 7项自动Qt链路检查：直接CSV导入、后台音频/共振峰、标无声、撤销、标静音、单文件/批量导出相同、导出后再编辑再导入恢复。使用真实Controller、QThread、按钮/QAction；替换文件选择对话框。**不是人工完整GUI验收**。
- Qt检查提前解析librosa.load以排除依赖冷启动对60秒超时的影响，结果JSON明确记录。
- 76条双路径评估记录中0个声学导出异常、0个无限值；6条20 ms音频的音高提取由于默认floor与最短窗口不兼容报错。这些异常保留，未算成通过。
- 打包与启动结果以本地 `build-final.log` / `packaged-startup.json` 为准；没有跑过的独立应用编辑导出操作不在验收范围。

## 复现与证据

先在项目根目录运行（音频/输出仅落到本地审计目录）：

```bash
venv/bin/python tests/validation/generate_signals.py output_validation/accuracy-audit/signals
PYTHONPATH=. venv/bin/python -m unittest discover -s tests/validation -p 'test_*.py' -v
venv/bin/python tests/validation/reference_checks.py
venv/bin/python tests/validation/evaluate_signals.py
QT_QPA_PLATFORM=offscreen venv/bin/python tests/validation/gui_export_check.py
```

独立Praat脚本默认使用macOS `/Applications/Praat.app/Contents/MacOS/Praat`，其他环境需调整路径。某些沙箱限制会使Praat CLI中止，不应误判为算法错误。测试使用标准库unittest而非计划中的pytest，避免为测试新增依赖。

证据位于 `output_validation/accuracy-audit/`：

- `baseline-contracts.log`、`baseline-edit.log`、`hf-before.log`、`schema-before.log`、`parameter-before.log`：修复前失败。
- `regression-final.log`：修复后自动回归。
- `reference/band_energy_boundary.json`：错误端点与独立FFT证据；早期同Praat脚本重复了错误端点，这一“通过”结论已经撤回。
- `reference/reference_results.json`、`reference_comparison.csv`：修复后的独立参考和版本差异。
- `signal-evaluation/evaluation.json`、`f0_errors.csv`、`pulse_floor_sensitivity.json`：合成/本地样本与脉冲参数敏感性。
- `gui/results-before-import-fix.json`、`gui/results.json`、`gui/annotator.png`：CSV重导入缺陷及修复后操作链路。
- `environment.json`：环境记录；每次专项脚本保留自己的源码hash，最终源码hash见 `final-source-hashes.json`。
- `docs/validation/metric-contract.csv`：50个字段的定义及状态，原审查行号可能因修复移动。

## 官方定义

- [Praat Jitter](https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html)：逐周期定义，通常用于持续元音。
- [Praat Shimmer](https://www.fon.hum.uva.nl/praat/manual/Voice_3__Shimmer.html)：相邻周期振幅变化。
- [Praat Burg共振峰](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html)：上限、阶数、窗口、预加重共同决定结果。

发布结论：此次尚未更新GitHub。先保留审计分支和证据；明确剩余指标的产品/研究口径后，才能发布完整声学特征的准确性声明。
