from pathlib import Path

import pandas as pd

from src.analytics import build_summary_table, category_distribution, dataset_summary
from src.exporter import export_reports
from src.reporting import build_executive_report, export_ml_report


def test_analytics_counts_are_computed_from_source_values():
    df = pd.DataFrame({
        'Drone_ID': [1, 1, 2],
        'Disaster_Type': [0, 1, 1],
        'Injury_Detection': [1, 0, 1],
        'Injury_Severity': [2, 0, 1],
        'Rescue_Priority_Score': [.2, .4, .6],
    })

    summary = dataset_summary(df)
    distribution = category_distribution(df, 'Disaster_Type')
    table = build_summary_table(df)

    assert summary['records'] == 3
    assert summary['drones'] == 2
    assert summary['injury_detection_code_1_records'] == 2
    assert summary['avg_priority'] == .4
    assert distribution['records'].sum() == 3
    assert 'Code 0' in table['value'].tolist()


def test_export_functions_write_reports_and_executive_summary(tmp_path):
    df = pd.DataFrame({'Disaster_Type': [0, 1], 'Rescue_Priority_Score': [.25, .75]})
    summary = pd.DataFrame({'metric': ['Disaster_Type'], 'value': ['Code 0'], 'count': [1]})
    files = export_reports(summary, df, tmp_path)
    ml_results = {
        'target': 'Injury_Severity',
        'feature_sets': {
            'baseline': {
                'results': {
                    'logistic_regression': {
                        'accuracy': .5,
                        'balanced_accuracy': .4,
                        'macro_f1': .35,
                        'feature_count': 4,
                    },
                },
            },
        },
    }
    ml_file = export_ml_report(ml_results, tmp_path / 'ml.csv')
    text = build_executive_report(
        {'records': 2, 'drones': 1, 'disaster_types': 2, 'avg_priority': .5},
        {'row_count': 2, 'duplicate_rows': 0, 'missing_cells': 0},
        {'processed_rows': 2, 'duplicate_rows_removed': 0},
        ml_results,
    )

    assert len(files) == 2
    assert all(Path(file).exists() for file in files)
    assert ml_file.exists()
    assert 'balanced_accuracy=0.4000' in text
    assert 'historical and not a live operational feed' in text
