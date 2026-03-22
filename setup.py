"""PyCoreML build — compiles Swift source into a Python-loadable .so"""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext


class SwiftBuildExt(build_ext):
    """Custom build_ext that calls `swift build` to compile the Swift extension."""

    def run(self):
        if sys.platform != "darwin":
            raise RuntimeError("pycoreml only supports macOS")

        swift_dir = Path(__file__).parent / "swift"
        pkg_config_path = sysconfig.get_config_var("LIBPC") or ""

        env = os.environ.copy()
        env["PKG_CONFIG_PATH"] = pkg_config_path

        print("🔨 Building Swift extension...")
        subprocess.check_call(
            ["swift", "build"],
            cwd=swift_dir,
            env=env,
        )

        build_dir = swift_dir / ".build" / "debug"
        dylib = build_dir / "libPyCoreML.dylib"
        if not dylib.exists():
            raise RuntimeError(f"Build succeeded but {dylib} not found")

        dest = Path(__file__).parent / "pycoreml" / "pycoreml.so"
        print(f"📦 Installing {dylib.name} → {dest}")
        import shutil
        shutil.copy2(dylib, dest)

    def get_ext_filename(self, ext_name):
        return ext_name + ".so"


setup(cmdclass={"build_ext": SwiftBuildExt})
