# PitchAnnotator 使用说明

## 1. 这个工具是做什么的

PitchAnnotator 用来查看、校正和导出音高轨迹。

它适合这些场景：

- 手动检查自动提取到的 F0 是否合理
- 把某些区间改成静音、清音或浊音
- 对明显错误的 pitch 点做人工修正
- 导出 Praat 风格 pitch 文件和声学特征表

## 2. 启动方式

### 2.1 打包版

如果你拿到的是已经打包好的版本：

- macOS：双击 `PitchAnnotator.app`
- Windows：打开 `PitchAnnotator` 文件夹，双击 `PitchAnnotator.exe`

### 2.2 源码版

如果你拿到的是源码文件夹：

- macOS：双击 `Start_PitchAnnotator.command`
- Windows：双击 `Start_PitchAnnotator.bat`

或者按命令行方式启动：

macOS:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Windows:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

建议 Python 版本：

- Python 3.11 或 3.12

项目根目录下已经准备了一个 `output/` 文件夹，建议把导出的 pitch csv、Praat Pitch 和 acoustic features 都统一放到这里。

## 3. 打开音频

在菜单栏选择：

`File -> Import Audio Files...`

支持的常见格式包括：

- `.wav`
- `.mp3`
- `.m4a`
- `.flac`
- `.aiff`
- `.aif`
- `.ogg`

打开后，界面会自动：

- 显示整段音频的频谱图
- 提取初始音高轨迹
- 显示顶部 segment 条带
- 显示 F0 20%、50%、80% 分位线
- 在左侧文件列表中加入该音频

如果一次导入多个音频：

- 左侧会显示文件列表
- 可以直接点击列表切换
- 也可以按键盘 `Up / Down` 切换
- 切换后会保留该音频之前的手动编辑状态

## 4. 界面怎么看

### 4.1 中央频谱图

- 横轴：时间
- 左侧纵轴：频率 `Hz`
- 右侧纵轴：音高 `semitone (st)`
- 灰度背景：频谱图
- 蓝色曲线：当前 F0 轨迹
- 红 / 橙 / 黄点：`F1 / F2 / F3`
- 蓝色高亮框：当前选择的时间段

### 4.2 顶部图例

顶部会显示两组图例。

第一组是 segment 颜色：

- `Silence`：静音区间
- `Voiceless`：清音区间
- `Voiced`：浊音区间

第二组是 F0 分位线：

- 蓝线：F0 20%
- 红线：F0 50%
- 绿线：F0 80%

### 4.3 状态栏

底部状态栏会显示：

- `Total`：当前音频总时长
- `Selection`：当前框选时长
- `F0 20% / 50% / 80%`，同时显示 `Hz` 和 `st`
- `Voice%`

其中 `Voice%` 的定义与 `Acoustic_analysis` 保持一致：

`voiced / (voiced + unvoiced)`

也就是说：

- 静音帧不计入分母
- 只在活动语音段中统计浊音比例

## 5. 选择时间段

把鼠标移动到频谱图中间，你会看到 `I` 形光标。

### 方法

1. 按住鼠标左键
2. 沿时间轴横向拖动
3. 松开鼠标

此时会出现蓝色高亮选区。

这个交互更接近视频编辑软件的时间段选择。

## 6. 播放功能

### 6.1 播放原始音频选区

先框选一个时间段，然后按：

`Space`

会播放该时间段的原始音频。

### 6.2 播放提取到的音高轨迹

先框选一个时间段，然后按：

`Shift + Space`

会播放该时间段的“F0 轨迹合成音”，用于和原始音频对照。

这个功能特别适合：

- 听一听自动 pitch 提取得对不对
- 判断某个地方是不是倍频 / 半频错误
- 检查人工修改后是否更合理

### 6.3 调节音量

右侧 `Playback` 区域有音量滑块，可以调整播放音量。

### 6.4 选择输出设备

右侧 `Playback` 区域可以选择 `Output Device`。

如果按空格播放没有声音，请先检查：

- 音量不是 `0%`
- 输出设备是否选对

## 7. 手动编辑音高

### 7.1 增加一个音高点

按住：

`Alt + 左键`

点击频谱图，程序会在最近谱峰处增加一个 pitch 点。

### 7.2 删除一个音高点

按住：

`Alt + Shift + 左键`

点击目标点附近，会把最近的一个点删掉，变成 unvoiced。

### 7.3 选择并拖拽一个音高点

直接左键点击一个已有的蓝色点，可以选中它。

然后拖拽这个点，就可以修改它的频率位置。

### 7.4 平移一整段选中的音高轨迹

1. 先框选一个时间区间
2. 程序会把这个区间内的有效 pitch 点高亮出来
3. 把鼠标放到这些高亮点上
4. 按住 `Shift`
5. 左键上下拖拽

这样会把该区间内所有高亮的 pitch 点整体上下平移。

### 7.5 撤销

按：

`Ctrl + Z`

可以撤销上一步编辑。

当前支持撤销：

- 设为 voiced
- 设为 unvoiced
- 设为 silence
- 增点
- 删点
- 拖拽改点

## 8. 设定区间类型

在右侧 `Manual Editing Tools` 区域，可以对当前选区进行标注。

### 8.1 Set Region to Voiced

把当前选区设为浊音。

程序会尽量在该区间内估计一个合理的 F0，而不是简单填一个常数。

### 8.2 Set Region to Unvoiced

把当前选区设为清音。

对应区间的 pitch 会被设为无声调值，但仍然属于活动语音。

### 8.3 Set Region to Silence

把当前选区设为静音。

这个区间：

- pitch 会被清空
- segment 会标记为 `Silence`
- `Voice%` 分母不会包含这段时间

## 9. 右侧 Pitch Parameters

右侧参数区目前遵循 Praat 默认思路。

默认关键参数包括：

- `Pitch Floor`
- `Pitch Top`
- `Voicing Threshold`
- `Silence Threshold`
- `Octave Cost`
- `Octave Jump Cost`
- `Voiced/Unvoiced Cost`

其中：

- 文件名中的 `gender2`、`SP`、`NV` 会自动调整 `Pitch Floor` 和 `Pitch Top`
- 其他 5 个参数保持 Praat 默认值

如果你修改了这些参数，点击：

`Recompute Initial Pitch`

程序会重新提取整段音高。

## 10. 导出功能

在菜单 `File` 中可以导出三类结果。

### 10.1 Export CSV...

导出当前 pitch 轨迹为 CSV。

内容包括：

- 音频文件名
- pitch 提取参数
- 时间
- 频率
- segment label

### 10.2 Export Batch Pitch CSVs...

如果左侧已经导入了多个音频，可以一次性批量导出所有音频的 pitch 轨迹 csv。

使用方法：

1. 在菜单中选择 `File -> Export Batch Pitch CSVs...`
2. 选择一个输出目录
3. 程序会自动在该目录下创建一个 `pitch_csv` 文件夹
4. 每个音频会导出一个单独的 csv，文件名格式为：

- `音频名_pitch.csv`

每个 csv 中会包含：

- 音频文件名
- pitch 提取参数
- 时间
- 频率
- segment label

### 10.3 Export Praat .Pitch...

导出为 Praat `.Pitch` 文件，便于在 Praat 或其他工具中继续使用。

### 10.4 Export Acoustic Features CSV...

导出声学特征 CSV。

特征口径参考 `Acoustic_analysis` 中的实现，包括：

- Voice%
- 强度统计
- F0 统计
- Jitter / Shimmer / HNR
- Formants
- 高频能量比例
- 谱斜率等

如果你手动修改了：

- `Silence / Voiceless / Voiced` 区间
- pitch 点
- 选区内整体平移后的 F0

那么导出时会尽量按你最终编辑后的状态重算相关指标。

### 10.5 Export Batch Acoustic Features CSV...

如果左侧已经导入了多个音频，可以导出一个批量 CSV。

这个文件会包含：

- 每个音频一行
- 每行对应一个音频最终编辑后的 acoustic features
- 同时附带该音频使用的 pitch 提取参数

导出时会弹出一个可编辑的文件名输入框。

如果你不修改文件名，程序会尝试根据第一个导入音频的文件名自动生成，例如：

- `sub1_SP_acoustic_features.csv`

如果没有识别出 `sub*` 和 `SP/NV` 信息，则默认使用：

- `batch_acoustic_features.csv`

### 10.6 Export Batch All...

如果你想一次同时导出：

- 所有音频各自的 pitch csv
- 一个总的 batch acoustic features csv

可以使用：

`File -> Export Batch All...`

流程是：

1. 选择一个输出目录
2. 程序会在该目录下：
   - 创建或复用 `pitch_csv` 文件夹
   - 输出所有音频各自的 `*_pitch.csv`
   - 同时生成一个你可编辑命名的 acoustic features 总表

在这一步里，程序会先弹出一个可编辑的 acoustic features 文件名输入框。

如果你不修改，默认文件名会尽量根据第一个音频自动生成，例如：

- `sub1_SP_acoustic_features.csv`

快捷键：

`Ctrl + Alt + Shift + E`

### 10.7 Export All...

快捷键：

`Ctrl + Shift + E`

会一键导出：

- `*_pitch.csv`
- `*.Pitch`
- `*_acoustic_features.csv`

默认文件名会继承当前音频文件名。

## 11. 常见问题

### 11.1 为什么看不到颜色条

顶部有 segment 时间条和图例：

- 黄色：静音
- 灰色：清音
- 绿色：浊音

如果整段音频几乎全是同一种状态，颜色条会比较单一，这是正常现象。

### 11.2 为什么 Voice% 没有想象中那么高

因为 Voice% 不是用整段录音长度算的，而是：

`voiced / (voiced + unvoiced)`

静音不进入分母。

### 11.3 为什么我设成 voiced 之后数值变化不大

可能原因包括：

- 你选中的区间很短
- 原来该区间就已经是 voiced
- 对整体分位数影响不大

但数据本身已经会更新，并且可用 `Ctrl + Z` 撤销。

### 11.4 macOS 打不开 app

如果第一次打开被系统拦截：

1. 右键 `PitchAnnotator.app`
2. 选择“打开”
3. 再次确认

### 11.5 Windows 下双击 exe 没反应

请确保你打开的是整个打包后的文件夹，而不是把单个 `.exe` 单独拷出来运行。

## 12. 推荐使用流程

给研究助理的推荐流程：

1. 打开音频
2. 检查自动提取到的 pitch
3. 框选问题区间
4. 用 `Space` 听原音
5. 用 `Shift + Space` 听 pitch 轨迹
6. 通过加点、删点、拖点、整段平移或改 segment 修正
7. 左侧切换到下一条音频继续检查
8. 最后导出 pitch CSV / Batch Pitch CSVs / Praat Pitch / Acoustic Features CSV / Batch Acoustic Features CSV / Batch All
