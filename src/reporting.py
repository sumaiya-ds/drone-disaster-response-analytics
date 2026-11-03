from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def build_quality_report_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        'rows': [len(df)],
        'columns': [len(df.columns)],
        'duplicate_rows': [int(df.duplicated().sum())],
        'missing_cells': [int(df.isna().sum().sum())],
        'null_columns': [int(df.isna().sum().astype(bool).sum())],
    }
    return pd.DataFrame(summary)


def export_summary_report(df: pd.DataFrame, output_path: str | Path) -> Path:
    summary = df.describe(include='all').T
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output)
    return output


def export_ml_report(model_results: Dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature_set, feature_data in model_results['feature_sets'].items():
        for model_name, data in feature_data['results'].items():
            rows.append({
                'target': model_results['target'],
                'feature_set': feature_set,
                'model': model_name,
                'accuracy': data['accuracy'],
                'balanced_accuracy': data['balanced_accuracy'],
                'macro_f1': data['macro_f1'],
                'feature_count': data['feature_count'],
            })
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def build_executive_report(
    analytics_summary: dict[str, Any],
    quality_report: dict[str, Any],
    pipeline_summary: dict[str, Any],
    model_results: dict[str, Any] | None = None,
) -> str:
    average_priority = analytics_summary.get('avg_priority')
    priority_text = f'{average_priority:.3f}' if average_priority is not None else 'Not available'
    lines = [
        'DRONE DISASTER ANALYSIS',
        'Assessment & Rescue Analytics — executive summary',
        '',
        'DATASET PROFILE',
        f"Processed records: {analytics_summary.get('records', 0):,}",
        f"Unique drone IDs: {analytics_summary.get('drones', 0):,}",
        f"Disaster source categories: {analytics_summary.get('disaster_types', 0):,}",
        f'Mean rescue-priority score: {priority_text}',
        '',
        'DATA QUALITY',
        f"Source records: {quality_report.get('row_count', 0):,}",
        f"Exact duplicate rows: {quality_report.get('duplicate_rows', 0):,}",
        f"Missing cells: {quality_report.get('missing_cells', 0):,}",
        f"Invalid timestamps: {quality_report.get('invalid_timestamp_count', 0):,}",
        f"Invalid coordinates: {quality_report.get('invalid_coordinate_count', 0):,}",
        '',
        'PIPELINE',
        f"Duplicate rows removed: {pipeline_summary.get('duplicate_rows_removed', 0):,}",
        f"Processed records: {pipeline_summary.get('processed_rows', analytics_summary.get('records', 0)):,}",
        f"Normalized fields: {', '.join(pipeline_summary.get('normalized_fields', [])) or 'None'}",
        'Missing values imputed: 0 (missing observations are preserved)',
    ]
    if model_results:
        lines.extend([
            '',
            f"MODEL EVALUATION — target: {model_results['target']}",
            'Stratified random holdout; results are exploratory and not deployment evidence.',
        ])
        for feature_set, feature_data in model_results['feature_sets'].items():
            for model_name, metrics in feature_data['results'].items():
                lines.append(
                    f"{feature_set} / {model_name}: accuracy={metrics['accuracy']:.4f}, "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                    f"macro_f1={metrics['macro_f1']:.4f}"
                )
    lines.extend([
        '',
        'LIMITATIONS',
        'The supplied data is historical and not a live operational feed; authenticity is not independently verified.',
        'Numeric category codes have no accompanying descriptive codebook.',
        'Geographic coordinates and medical measures are not independently validated.',
    ])
    return '\n'.join(lines) + '\n'
