"""Docker build helper: install, without any download, every pinned package that a local wheelhouse holds.

    python use_wheelhouse.py constraints.txt /wheels

pip treats a wheel in --find-links and the same wheel on an index as equals and may fetch it from the network, so a
local wheelhouse is installed first, explicitly and with --no-index; the normal `pip install -r constraints.txt`
that follows then finds those pins satisfied and only downloads what the wheelhouse lacks. An empty or missing
wheelhouse (the default) does nothing.
"""
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main(constraints: str, wheels: str) -> None:
    pins = {}
    for line in Path(constraints).read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)==(\S+)$", line.strip())
        if m:
            pins[norm(m.group(1))] = m.group(2)
    chosen = []
    for wheel in sorted(Path(wheels).glob("*.whl")) if Path(wheels).is_dir() else []:
        name, version = unquote(wheel.name).split("-")[:2]
        if pins.get(norm(name)) == version:
            chosen.append(str(wheel))
    if not chosen:
        print("no local wheels match the pins; everything is downloaded")
        return
    print(f"installing {len(chosen)} pinned packages from {wheels} without downloading")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *chosen], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
