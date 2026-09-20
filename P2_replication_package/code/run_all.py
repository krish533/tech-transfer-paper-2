from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    analysis_script = package_root / "code" / "replication.py"
    log_path = package_root / "paper_outputs" / "logs" / "replication_output.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    analysis = subprocess.run(
        [sys.executable, str(analysis_script)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(analysis.stdout, end="")
    log_path.write_text(analysis.stdout, encoding="utf-8", newline="")
    if analysis.returncode:
        raise subprocess.CalledProcessError(
            analysis.returncode,
            [sys.executable, str(analysis_script)],
        )

    print(
        "Replication package run completed. Log: "
        f"{log_path.relative_to(package_root).as_posix()}"
    )


if __name__ == "__main__":
    main()
