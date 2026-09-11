"""Reject malformed observation options before touching GPU or output files."""
import pathlib
import subprocess
import sys
import tempfile

exe = sys.argv[1]
with tempfile.TemporaryDirectory(prefix="pf-velocity-cli-") as directory:
    root = pathlib.Path(directory)
    cases = [
        (["--velocity-moments", ""], "velocity recording requires"),
        (["--velocity-moments", "v.bin"], "velocity recording requires"),
        (["--velocity-moments-reference"], "velocity recording requires"),
        (["--velocity-spatial-stride", "100"], "velocity recording requires"),
        (["--velocity-spatial-stride", "0"], "expected an integer"),
        (["--velocity-spatial-stride", "1001"], "expected an integer"),
        (["--velocity-dense-start", "-1"], "expected an integer"),
        (["--out", "same", "--velocity-moments", "./same"],
         "output files must use distinct paths"),
        (["--out", "t.txt", "--velocity-moments", "same",
          "--boundary-out", "./same", "--boundary-interval", "1000"],
         "output files must use distinct paths"),
        (["--out", "t.txt", "--velocity-moments", "checkpoint.bin",
          "--checkpoint-dir", "."], "output files must use distinct paths"),
    ]
    for flags, diagnostic in cases:
        result = subprocess.run([exe, *flags], cwd=root, capture_output=True, text=True)
        assert result.returncode == 2, (flags, result.stdout, result.stderr)
        assert diagnostic in result.stderr, (flags, result.stderr)
    assert not list(root.iterdir()), "invalid options created output files"
print("PASS: observation option validation and output aliases")
