from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def export_reports(summary_table: pd.DataFrame, cleaned_df: pd.DataFrame, output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    csv_path = output / 'analytics_summary.csv'
    summary_table.to_csv(csv_path, index=False)
    files.append(str(csv_path))

    cleaned_path = output / 'cleaned_disaster_dataset.csv'
    cleaned_df.to_csv(cleaned_path, index=False)
    files.append(str(cleaned_path))
    return files
