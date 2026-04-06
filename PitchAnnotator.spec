# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata


ROOT = Path(SPEC).resolve().parent

datas = []
binaries = []
hiddenimports = [
    "soundfile",
    "scipy.special.cython_special",
]

datas += copy_metadata("librosa")

for package_name in ("praat-parselmouth", "pyqtgraph", "librosa"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PitchAnnotator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PitchAnnotator",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PitchAnnotator.app",
        icon=None,
        bundle_identifier="com.pitchannotator.app",
    )
