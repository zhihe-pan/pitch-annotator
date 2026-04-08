# PitchAnnotator

PitchAnnotator 是一个用于手动校正和检查音高轨迹的桌面工具，适合语音、情感发声、声学分析等实验场景。

它目前支持：

- 批量载入常见音频文件并在左侧列表中切换
- 显示 Praat 风格频谱图、F0 轨迹、F1/F2/F3 共振峰
- 使用 Praat 风格参数提取 F0
- 手动增加、删除、拖拽单个音高点
- 框选一个时间区间后整体上下平移该区间内的音高轨迹
- 将选区标记为 `Silence / Voiceless / Voiced`
- 播放原始选区音频
- 播放提取到的音高轨迹，便于人工核对
- 导出 pitch CSV、批量 pitch CSV、Praat `.Pitch` 文件、单文件和批量声学特征 CSV
- 一键批量导出所有 pitch CSV 和 batch acoustic features CSV

详细使用说明见 [USAGE_zh-CN.md](/Users/caddice/Desktop/Acoustic/USAGE_zh-CN.md)。

## 给研究助理的推荐使用方式

如果你准备把源码直接发给研究助理，建议发一个干净的源码压缩包，而不是整个项目目录。

压缩包中建议保留：

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

不要包含这些内容：

- `venv/`
- `build/`
- `dist/`
- `release/`
- `.git/`
- `__pycache__/`
- 临时导出的 csv
- 与程序运行无关的大型测试音频

## 从源码运行

如果需要从源码启动：

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Windows 下：

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

也可以直接双击：

- macOS: `Start_PitchAnnotator.command`
- Windows: `Start_PitchAnnotator.bat`

建议的 Python 版本：

- Python 3.11 或 3.12

## 研究助理快速上手

1. 解压源码压缩包
2. 安装 Python 3.11 或 3.12
3. 按上面的命令创建虚拟环境并安装依赖
4. 运行 `python main.py`
5. 在菜单中选择 `File -> Import Audio Files...`
6. 导入一个或多个音频后，在左侧列表中切换检查
7. 编辑完成后导出单文件 pitch CSV、批量 pitch CSV、或批量 acoustic features CSV

建议研究助理把所有导出结果统一放在项目根目录下的 `output/` 文件夹中。

## 本地打包

### macOS 或 Windows

请在对应系统上本地执行：

```bash
python -m pip install -r requirements.txt
python build.py
```

构建完成后会生成：

- `dist/`：PyInstaller 原始输出
- `release/`：适合分发的压缩包

例如：

- `release/PitchAnnotator-macos.zip`
- `release/PitchAnnotator-windows.zip`

注意：

- macOS 应用必须在 macOS 上打包
- Windows 应用必须在 Windows 上打包
- 一般不要在一个系统上交叉打包另一个系统的桌面应用

## GitHub Actions 自动构建

仓库已经包含自动构建工作流：

- 文件位置：[.github/workflows/build-release.yml](/Users/caddice/Desktop/Acoustic/.github/workflows/build-release.yml)
- 触发条件：
  - 推送到 `main`
  - Pull Request
  - 手动触发 `workflow_dispatch`

工作流会在：

- `macos-latest`
- `windows-latest`

上分别构建，并上传 zip 产物到 GitHub Actions 的 Artifacts。

## 维护者说明

如果你准备长期把这个工具分享给实验室成员，推荐流程是：

1. 代码推到 GitHub
2. 研究助理优先从源码运行
3. 如果后面需要真正开箱即用，再用 GitHub Actions 自动构建 macOS 和 Windows 版本
4. 文档和快捷键变更时，同时更新 [USAGE_zh-CN.md](/Users/caddice/Desktop/Acoustic/USAGE_zh-CN.md)
