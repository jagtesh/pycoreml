"""PyCoreML build — compiles Swift source into a Python-loadable .so"""
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class SwiftBuildExt(build_ext):
    """Custom build_ext that calls `swift build` to compile the Swift extension."""

    def run(self):
        if sys.platform != "darwin":
            raise RuntimeError("pycoreml only supports macOS")

        src_dir = Path(__file__).parent
        swift_dir = src_dir / "swift"
        pkg_config_path = sysconfig.get_config_var("LIBPC") or ""

        env = os.environ.copy()
        env["PKG_CONFIG_PATH"] = pkg_config_path

        print("🔨 Building Swift extension...")
        subprocess.check_call(
            ["swift", "build"],
            cwd=swift_dir,
            env=env,
        )

        # Find the built .dylib (check platform-specific and legacy paths)
        lib_name = "libPyCoreML.dylib"
        candidates = [
            swift_dir / ".build" / "debug" / lib_name,
            swift_dir / ".build" / "arm64-apple-macosx" / "debug" / lib_name,
        ]
        dylib = next((p for p in candidates if p.exists()), None)
        if dylib is None:
            raise RuntimeError(f"Build succeeded but {lib_name} not found in .build/")

        # Copy to _native/ subdirectory (avoids shadowing the package)
        dest_dir = src_dir / "pycoreml" / "_native"
        dest_dir.mkdir(exist_ok=True)
        src_dest = dest_dir / "pycoreml.so"
        print(f"📦 Installing {dylib.name} → {src_dest}")
        shutil.copy2(dylib, src_dest)

        # Also copy to build_lib (for regular pip install / wheel builds)
        if self.build_lib:
            build_dest = Path(self.build_lib) / "pycoreml" / "_native" / "pycoreml.so"
            build_dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"📦 Installing {dylib.name} → {build_dest}")
            shutil.copy2(dylib, build_dest)


# Dummy extension so setuptools invokes build_ext
setup(
    ext_modules=[Extension("pycoreml._swift", sources=[])],
    cmdclass={"build_ext": SwiftBuildExt},
)
