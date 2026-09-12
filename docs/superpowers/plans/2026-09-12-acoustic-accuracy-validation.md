# 声学特征准确性验证计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每个导出指标提供可复现的正确性证据、适用范围及未解决限制，而非承诺任意音频上的绝对准确。

**Architecture:** 冻结当前版本，用独立公式、独立 Praat 脚本和已知真值合成材料三类证据交叉检查。将数值实现正确性、音频估计误差和人工编辑到导出的完整链路分别报告。

**Tech Stack:** 当前 venv、Python、NumPy、Parselmouth、独立 Praat CLI、pytest、CSV/JSON 审计记录。

**Spec:** 本次用户请求：计划验证工具准确性；本文包含验证范围和验收设计。

## 全局约束

- 当前只制定计划，不修改算法或推送发布。
- 固定基线提交 039e1c5；执行时记录实际 commit、Python、NumPy、Parselmouth 及其内嵌 Praat 和独立 Praat 版本。
- 本地 stash、check_devices.py、output_validation/ 不改动；真实研究音频和派生数据不提交到 GitHub。
- 参考脚本不得调用本项目的计算函数。相同 Praat 算法的结果一致只证明集成一致，不能作为独立生理真值。
- 先冻结指标定义及容差，再运行比较；发现失败不得通过放宽阈值消除。版本/参数不一致单列诊断。
- 修复必须有能复现问题的测试，验证修复后再运行相关回归。可能改变研究口径的修复先形成具体方案供用户判断。

## 文件与产物

- docs/validation/metric-contract.csv：每个导出字段的定义、单位、公式、分母、帧筛选、聚合方式、缺失值规则、代码位置、参考来源。
- tests/validation/generate_signals.py：固定种子合成材料和真值 JSON。
- tests/validation/reference_features.praat：独立 Praat 原始帧/脉冲及指标导出。
- tests/validation/test_metric_contracts.py：独立公式及边界检查。
- tests/validation/test_reference_agreement.py：逐字段对照。
- tests/validation/test_edit_export.py：人工编辑和 CSV 往返链路。
- docs/validation/accuracy-report.md：结果、失败、适用范围和发布结论。
- 本机验证输出写入 output_validation/accuracy-audit/；参考输出记录输入 SHA256、参数和版本。

## 任务 1：冻结指标定义与调用路径

- [ ] 枚举 backend/acoustic_analysis.py 的基础结果与 backend/acoustic_features.py 的覆盖结果，涵盖 F0、时长与比例、强度、音质、频谱、共振峰全部字段。
- [ ] 对照 backend/audio_core.py、core/exporter.py、main.py，列出自动提取、CSV 导入、编辑后导出、单文件与批量路径。
- [ ] 为每项填写口径表；重点查 Voiced_percent 是 0–1 还是 0–100、有效时间与总时间分母、SD 的 ddof、半音参考频率、强度 dB 均值方式、频谱幅度/功率加权和 slope 单位。
- [ ] 核对文件名决定参数（gender2/SP/NV）、八度校正与手工 F0 的优先级、异常返回 NaN 是否掩盖程序错误。
- [ ] 把争议口径单独列出；不把当前代码自身当成规范。产物：完整字段合同和优先问题表。

## 任务 2：建立可复现测试材料

- [ ] 生成 100/200/400 Hz 稳态谐波音、连续升降调、八度阶跃、已知脉冲周期和幅度扰动信号、已知共振峰源滤波元音。
- [ ] 每种主信号制作全有声、含首尾静音、两段有声中间无声、0.02 秒短音频版本；补充全静音、纯噪声、幅度缩放 0.5/2、单声道和双声道、16/44.1/48 kHz。
- [ ] CSV 边界包括空表、空值、NaN/Inf、负时间、越界时间、乱序/重复时间、非均匀时间步、标签与 F0 矛盾、音频重名及路径失效。
- [ ] 真实音频先筛选约 24 条：持续元音、连续语音、非言语发声各 8 条，覆盖不同 F0 和噪声；材料不足明确标记，不虚构覆盖。真实材料只在本地使用。
- [ ] 真值随信号一起保存；Jitter/Shimmer 使用已知周期/振幅序列推导参考，不能把 F0 帧差当作逐周期真值。

## 任务 3：优先验证高风险计算

- [ ] 对相同有声片段分别比较原时间轴计算、逐段计算后按定义聚合、当前拼接后计算，定位接缝对 Jitter/Shimmer/HNR、频谱和共振峰的影响。不能仅与同样拼接的参考程序比较。
- [ ] 用独立 Praat 脚本输出脉冲和有效周期，显式设置 pitch floor/ceiling、period limits、maximum period/amplitude factors、HNR 参数。
- [ ] 共振峰核对 ceiling、窗长、预加重、时间步、有效帧和带宽过滤；同一音频改文件名进行敏感性试验，区分显式参数变化与意外行为。
- [ ] F0 检查帧时间对齐、插值、八度跳变与有声判定；逐帧评估和汇总指标分开报告。
- [ ] 对合成音先用真值评估；真实音频与 Praat 做同参数对照，同时保留人工核查的不确定性。

## 任务 4：验收规则与数值比较

- [ ] 独立公式计算的同一帧数组：初始工程容差 abs <= 1e-10 + 1e-8 * abs(reference)；标签、计数、字段集合和 NaN 位置必须完全一致。
- [ ] 时间长度和边界另行检查：离散帧区间约定一致时应相符；允许的离散误差上限为一个帧步长，但必须解释来源，不能掩盖越界和重复计数。
- [ ] 同参数同 Praat 版本的原始输出先以 abs <= 1e-8 + 1e-6 * abs(reference) 排查集成误差；不同版本的差异独立记录，不能自动判通过。
- [ ] 上述容差是数值一致性门槛，不是科研有效性阈值。真值估计报告 F0 cents 误差、超过 20% 的 gross error 比例、有声判定 precision/recall、F1–F3 Hz 误差、音质和频谱指标绝对及相对偏差。
- [ ] 在任务 1 完成后、测试结果揭晓前，依据具体指标用途与引用依据冻结估计误差验收门槛；没有依据的指标只能报告误差和适用范围，不签发“准确”结论。
- [ ] 每个字段输出 reference、actual、absolute_error、relative_error、NaN_match、input_id、parameter_set、status。近零参考值优先看绝对误差；不能用高相关代替一致性。

## 任务 5：编辑、导入和导出链路

- [ ] 自动提取后分别修改单点 F0、整体倍频/半频、改变有声/无声/静音标签、撤销；按字段合同检查应变化和应不变的指标。
- [ ] Pitch CSV 导出后直接重新导入，核对轨迹、标签、参数、原音频匹配和重新导出的特征；容差明确考虑 CSV 的六位小数序列化。
- [ ] 同一材料单文件/批量导出一致，切换文件不串参数；保留首尾区间和不规则时间轴案例。
- [ ] 执行 GUI 实际操作并检查结果文件；初始化成功不能替代交互验收。
- [ ] 源码版验证后，再构建独立应用验证音频读取、导入、编辑和声学导出；打包未运行就报告未验证。

## 任务 6：问题修复与交付

- [ ] 每个失败定位到输入、参数、帧/脉冲、聚合或导出层；保存最小复现与修复前失败证据。
- [ ] 对明确程序错误添加回归后修复；改变分析口径的方案先说明旧值/新值和研究影响。
- [ ] 相关检查和完整导出回归通过后，以专门提交记录修复；不混入实验字段。
- [ ] 报告按指标标记：已验证（限定条件）、受限可用、未通过、未验证；列出输入范围、环境、误差、剩余风险。
- [ ] 仅在执行和发布获授权后更新 GitHub；用户文档只写使用条件、参数与解释边界，不写内部迁移过程。

## 参考标准

- https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html
- https://www.fon.hum.uva.nl/praat/manual/Voice_3__Shimmer.html
- https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html
- 其他指标在任务 1 查询对应官方定义并记录具体页面与算法版本。

## 完成条件

所有导出字段均有合同和状态；高风险问题全部有结论或明确阻断；原始对照可复现；测试没有把项目函数当参考；报告明确源码/GUI/独立应用各自验证程度。不得把“与旧版本相同”或“与 Praat 相同”写成对任意研究材料的准确性保证。

## 2026-09-12 执行记录

用户已授权执行本计划。实际结果见 `docs/validation/accuracy-report.md`。已完成字段审查、36个合成音双路径与两个本地音频案例、独立参考、28项回归、7项自动Qt链路，以及明确程序错误修复。真实语料不足24条，完整人工GUI/打包交互验收和若干指标的科学有效性仍未完成；因此没有整体准确性通过结论。分段音质聚合方式待用户选择，未擅自改变。测试框架使用已有标准库unittest。
