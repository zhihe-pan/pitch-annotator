# PitchAnnotator

PitchAnnotator 是一个用于手动校正和检查音高轨迹的桌面工具，适合语音、情感发声、声学分析等实验场景。

它目前支持：

- 载入常见音频文件并显示频谱图
- 使用 Praat 风格参数提取 F0
- 手动增加、删除、拖拽音高点
- 将选区标记为 `Silence / Voiceless / Voiced`
- 播放原始选区音频
- 播放提取到的音高轨迹，便于人工核对
- 导出 pitch CSV、Praat `.Pitch` 文件和声学特征 CSV

详细使用说明见 [USAGE_zh-CN.md](/Users/caddice/Desktop/Acoustic/USAGE_zh-CN.md)。

## 给研究助理的推荐使用方式

最推荐直接使用已经打包好的发布版：

- macOS：打开 `PitchAnnotator.app`
- Windows：打开 `PitchAnnotator` 文件夹中的 `PitchAnnotator.exe`

这样不需要自己安装 Python 环境，属于“开箱即用”方案。

## 开发环境运行

如果需要从源码启动：

```bash
python -m venv venv
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

## 分发建议

### 给 macOS 用户

推荐把以下文件打包发给对方：

- `PitchAnnotator.app`
- 本说明文档

第一次打开时，如果 macOS 提示安全限制，可按下面操作：

1. 在 Finder 中右键 `PitchAnnotator.app`
2. 选择“打开”
3. 再次确认打开

### 给 Windows 用户

推荐把整个 `PitchAnnotator` 文件夹完整发给对方，不要只发单个 `.exe`。

原因是 PyInstaller 的 onedir 模式会依赖旁边的动态库和资源文件。

## 维护者说明

如果你准备长期把这个工具分享给实验室成员，推荐流程是：

1. 代码推到 GitHub
2. 用 GitHub Actions 自动构建 macOS 和 Windows 版本
3. 在每次稳定版本后，把 `release/*.zip` 或 Actions Artifacts 发给研究助理
4. 文档和快捷键变更时，同时更新 [USAGE_zh-CN.md](/Users/caddice/Desktop/Acoustic/USAGE_zh-CN.md)
