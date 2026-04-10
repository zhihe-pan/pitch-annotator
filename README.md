# PitchAnnotator

PitchAnnotator 是一个用于人工校正音高轨迹（F0）的桌面工具，适用于语音和声学研究流程。

它支持：

- 批量导入音频并逐条检查
- 频谱图 + F0 + F1/F2/F3 联合可视化
- Praat 风格参数提取音高
- 手动加点、删点、拖点、区间整体平移
- 区间标注 `Silence / Voiceless / Voiced`
- 播放原始选区和 F0 合成轨迹
- 导出当前文件或批量文件的结果
  - `Export CSV...`
  - `Export Spectrogram Plot...`
  - `Export Batch Pitch CSVs...`
  - `Export Batch Spectrogram Plots...`
  - `Export Praat .Pitch...`
  - `Export Acoustic Features CSV...`
  - `Export Batch Acoustic Features CSV...`
  - `Export All...`
  - `Export Batch All...`

详细中文使用说明见 [USAGE_zh-CN.md](/Users/caddice/Desktop/Acoustic/USAGE_zh-CN.md)。

## 环境要求

- Python 3.11 或 3.12
- 依赖见 `requirements.txt`
- 建议安装 Praat（用于外部 Praat filtered autocorrelation 路径）

Windows 上如需稳定走外部 Praat，推荐：

1. 安装 Praat
2. 配置环境变量 `PRAAT_PATH` 指向 `Praat.exe` 或 `praatcon.exe`
3. 在应用状态栏确认 `Pitch source: External Praat filtered AC`

## 从源码运行

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

也可以直接双击：

- `Start_PitchAnnotator.command`（macOS）
- `Start_PitchAnnotator.bat`（Windows）

## 给研究助理的发放建议

建议发送“源码压缩包”，且仅包含运行所需内容：

- `main.py`
- `backend/`
- `core/`
- `ui/`
- `acoustic_analysis/`
- `requirements.txt`
- `README.md`
- `USAGE_zh-CN.md`
- `Start_PitchAnnotator.command`
- `Start_PitchAnnotator.bat`

建议排除：

- `.git/`
- `venv/`
- `build/` `dist/` `release/`
- `__pycache__/`
- 临时导出结果和无关测试音频

## 打包桌面版

在对应系统本机执行：

```bash
python -m pip install -r requirements.txt
python build.py
```

生成目录：

- `dist/`（PyInstaller 原始输出）
- `release/`（可分发压缩包）
