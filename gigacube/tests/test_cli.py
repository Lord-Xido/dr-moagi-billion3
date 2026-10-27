#!/usr/bin/env python3
"""Independent CLI smoke test for source-file exact reconstruction."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

ENGINE = Path(__file__).resolve().parents[1] / "jarvisx_gigacube.py"

def run(*args):
    result = subprocess.run(
        [sys.executable, str(ENGINE), *map(str, args)],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    with tempfile.TemporaryDirectory(prefix="gigacube_ci_") as tmp:
        root = Path(tmp)
        original = root / "sample.bin"
        database = root / "sample.sqlite"
        recovered = root / "recovered.bin"
        visual = root / "state.html"
        original.write_bytes(bytes(range(256)) * 9 + b"gigacube-boundary-test")
        baseline = sha(original)
        run("init", "--input", original, "--db", database, "--brick", 16)
        run("iterate", "--db", database, "--count", 4)
        state = json.loads(run("status", "--db", database))
        assert state["iteration"] == 4, state
        assert state["allocated_bricks"] >= 2, state
        run("render", "--db", database, "--output", visual, "--points", 128)
        assert visual.exists() and visual.stat().st_size > 1000
        result = json.loads(run("recover", "--db", database, "--output", recovered))
        assert result["exact_reconstruction"] is True, result
        assert sha(recovered) == baseline
        assert recovered.read_bytes() == original.read_bytes()
        post = json.loads(run("status", "--db", database))
        assert post["iteration"] == 4, post
        run("reverse", "--db", database, "--count", 4)
        zero = json.loads(run("status", "--db", database))
        assert zero["iteration"] == 0, zero
        print("PASS: CLI ingest / 4 forward cycles / 3D render / snapshot recovery / 4 reverse cycles")

if __name__ == "__main__":
    main()
