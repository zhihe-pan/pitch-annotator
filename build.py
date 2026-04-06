from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
RELEASE_DIR = ROOT / "release"
APP_NAME = "PitchAnnotator"


def _platform_slug() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def _pyinstaller_output() -> Path:
    if sys.platform == "darwin":
        return DIST_DIR / f"{APP_NAME}.app"
    return DIST_DIR / APP_NAME


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyInstaller is not installed in the current environment.\n"
            "Please run: python -m pip install -r requirements.txt"
        ) from exc


def clean() -> None:
    for path in (BUILD_DIR, DIST_DIR, RELEASE_DIR):
        if path.exists():
            shutil.rmtree(path)


def run_pyinstaller() -> None:
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(ROOT / "PitchAnnotator.spec")]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def package_release() -> Path:
    target = _pyinstaller_output()
    if not target.exists():
        raise FileNotFoundError(f"Expected build output not found: {target}")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    release_name = f"{APP_NAME}-{_platform_slug()}"
    staging_dir = RELEASE_DIR / release_name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    destination = staging_dir / target.name
    if target.is_dir():
        shutil.copytree(target, destination)
    else:
        shutil.copy2(target, destination)

    for extra_name in ("README.md", "USAGE_zh-CN.md", "Start_PitchAnnotator.command", "Start_PitchAnnotator.bat"):
        extra_path = ROOT / extra_name
        if extra_path.exists():
            shutil.copy2(extra_path, staging_dir / extra_path.name)

    archive_base = RELEASE_DIR / release_name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=RELEASE_DIR, base_dir=release_name)
    return Path(archive_path)


def main() -> None:
    print(f"Building {APP_NAME} for {_platform_slug()}...")
    ensure_pyinstaller()
    clean()
    run_pyinstaller()
    archive = package_release()
    print(f"Build completed successfully.\nRelease archive: {archive}")


if __name__ == "__main__":
    main()
