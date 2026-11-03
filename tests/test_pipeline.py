from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import (
    build_feature_matrix,
    clean_dataframe,
    load_and_validate_data,
    profile_dataframe,
    save_uploaded_dataset,
    validate_dataframe,
)
from config import DEFAULT_DATASET

ROOT = Path(__file__).resolve().parents[1]


def test_default_dataset_path_is_relative_to_the_project_location():
    assert DEFAULT_DATASET == ROOT / 'Autonomous_Medical_Aid_Dataset.csv'
    assert DEFAULT_DATASET.is_file()


def test_load_and_validate_data_builds_quality_profile():
    result = load_and_validate_data(ROOT / 'Autonomous_Medical_Aid_Dataset.csv')

    assert result['row_count'] == 44569
    assert result['quality_report']['schema']['valid']
    assert result['quality_report']['missing_cells'] == 0
    assert result['quality_report']['duplicate_rows'] == 0
    assert result['profile']['row_count'] == 44569
    assert 'Rescue_Priority_Score' in result['profile']['numeric_statistics']
    assert result['transformation_summary']['missing_values_imputed'] == 0
    summary = result['transformation_summary']
    assert len(summary['stage_log']) == 7
    assert all(stage['status'] == 'Complete' for stage in summary['stage_log'])
    assert all(stage['duration_seconds'] >= 0 for stage in summary['stage_log'])
    assert summary['processing_duration_seconds'] >= 0
    assert summary['rejected_rows'] == 0
    assert summary['source_file_size_bytes'] > 0
    assert summary['processed_file_size_bytes'] > 0
    assert summary['feature_file_size_bytes'] > 0
    engineered = pd.read_csv(result['feature_path'])
    assert len(engineered) == 44569
    assert {'hour', 'battery_risk_flag', 'hazard_risk_score', 'resource_pressure'} <= set(engineered.columns)


def test_schema_validation_reports_missing_required_fields():
    result = validate_dataframe(pd.DataFrame({'Timestamp': ['2024-01-01']}))

    assert result['schema']['valid'] is False
    assert result['schema']['missing_required_columns'] == ['Latitude', 'Longitude']


def test_validation_counts_out_of_range_and_unparseable_numeric_values():
    source = pd.DataFrame({
        'Timestamp': ['2024-01-01'],
        'Latitude': [24.0],
        'Longitude': [46.0],
        'Battery_Level_percent': [101],
        'Speed_kmph': ['not numeric'],
    })

    report = validate_dataframe(source)

    assert report['invalid_range_issues'] == {
        'Battery_Level_percent': 1,
        'Speed_kmph': 1,
    }


def test_cleaning_removes_duplicates_and_preserves_missing_values():
    source = pd.DataFrame({
        'Timestamp': ['2024-01-01', 'invalid', '2024-01-01'],
        'Latitude': ['12.5', 'bad', '12.5'],
        'Longitude': [45.0, None, 45.0],
        'measurement': [None, 5.0, None],
    })

    cleaned, summary = clean_dataframe(source)

    assert len(cleaned) == 2
    assert summary['duplicate_rows_removed'] == 1
    assert summary['missing_values_imputed'] == 0
    assert pd.isna(cleaned['Timestamp'].iloc[1])
    assert pd.isna(cleaned['Latitude'].iloc[1])
    assert pd.isna(cleaned['measurement'].iloc[0])


def test_profile_and_feature_engineering_are_data_derived():
    source = pd.DataFrame({
        'Timestamp': pd.to_datetime(['2024-01-01 02:00', '2024-01-01 20:00', None]),
        'Battery_Level_percent': [10.0, 60.0, None],
        'Latitude': [24.0, 25.0, 26.0],
        'Longitude': [46.0, 47.0, 48.0],
        'Proximity_to_Hazard_meters': [10.0, 20.0, 30.0],
        'Medical_Supply_Inventory': [4.0, 6.0, 8.0],
        'Nearby_Resources_Count': [1, 2, 3],
        'Weather_Condition': [0, 1, 2],
    })

    profile = profile_dataframe(source)
    features = build_feature_matrix(source)

    assert profile['row_count'] == 3
    assert profile['numeric_statistics']['Battery_Level_percent']['count'] == 2
    assert features['hour'].tolist()[:2] == [2, 20]
    assert features['is_night'].tolist()[:2] == [1, 1]
    assert pd.isna(features['is_night'].iloc[2])
    assert features['battery_risk_flag'].iloc[0] == 1
    assert 'resource_pressure' in features
    assert 'weather_risk' not in features


def test_uploaded_file_is_preserved_and_processed_separately(tmp_path):
    content = (
        b'Timestamp,Latitude,Longitude,metric\n'
        b'2024-01-01,24.0,46.0,1\n'
        b'2024-01-01,24.0,46.0,1\n'
    )
    raw_path = save_uploaded_dataset(content, '..\\sample.csv', tmp_path / 'raw')
    processed_path = tmp_path / 'processed' / 'sample.csv'
    result = load_and_validate_data(raw_path, output_path=processed_path)

    assert raw_path.name == 'sample.csv'
    assert raw_path.read_bytes() == content
    assert processed_path.exists()
    assert (tmp_path / 'processed' / 'sample_engineered_features.csv').exists()
    assert result['row_count'] == 1
    assert result['quality_report']['duplicate_rows'] == 1
    assert result['transformation_summary']['duplicate_rows_removed'] == 1
    assert result['transformation_summary']['rejected_rows'] == 0
    assert result['transformation_summary']['stage_log'][-1]['stage'] == 'Persist engineered dataset'


def test_invalid_upload_schema_is_rejected(tmp_path):
    bad_path = tmp_path / 'bad.csv'
    bad_path.write_text('Timestamp,Latitude\n2024-01-01,24.0\n', encoding='utf-8')

    with pytest.raises(ValueError, match='Longitude'):
        load_and_validate_data(bad_path, output_path=tmp_path / 'processed.csv')
