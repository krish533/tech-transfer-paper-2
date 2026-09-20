from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    pipeline_dir = package_root / "pipeline"
    log_path = package_root / "paper_outputs" / "logs" / "replication_output.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, str(pipeline_dir / "01_build_merged_panel.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(pipeline_dir / "02_validate_paper1_consistency.py")],
        check=True,
    )

    analysis = subprocess.run(
        [sys.executable, str(pipeline_dir / "03_run_analysis.py")],
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
            [sys.executable, str(pipeline_dir / "03_run_analysis.py")],
        )

    print(
        "Replication package run completed. Log: "
        f"{log_path.relative_to(package_root).as_posix()}"
    )


if __name__ == "__main__":
    main()
