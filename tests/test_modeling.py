import numpy as np
import pandas as pd
import pytest

from pages.ml_analytics import _comparison_table
import components.ui as model_ui
from src.modeling import (
    MODEL_RESULT_SCHEMA_VERSION,
    prepare_model_features,
    train_compare_models,
    validate_model_results,
)


def _sample_records(rows: int = 180) -> pd.DataFrame:
    random = np.random.default_rng(42)
    target = np.resize([0, 1, 2], rows)
    return pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=rows, freq='h'),
        'Latitude': random.normal(24, 1, rows),
        'Longitude': random.normal(46, 1, rows),
        'Weather_Condition': random.integers(0, 3, rows),
        'Disaster_Type': random.integers(0, 4, rows),
        'Injury_Type': random.integers(0, 3, rows),
        'Blood_Loss_Severity': random.integers(0, 3, rows),
        'Pain_Level_Score': random.normal(5, 1, rows),
        'Respiration_Rate_breaths_per_min': random.normal(16, 2, rows),
        'Heart_Rate_bpm': random.normal(80, 8, rows),
        'Recommended_Action': random.integers(0, 4, rows),
        'Rescue_Priority_Score': random.random(rows),
        'Injury_Detection': random.integers(0, 2, rows),
        'Injury_Severity': target,
    })


def test_target_is_never_used_as_a_feature_and_assessment_fields_are_audited():
    df = _sample_records()

    baseline, _, baseline_columns = prepare_model_features(df, 'Injury_Severity')
    conservative, _, conservative_columns = prepare_model_features(df, 'Injury_Severity', leakage_aware=True)

    assert 'Injury_Severity' not in baseline_columns
    assert 'Injury_Severity' not in conservative_columns
    assert 'Injury_Type' in baseline
    assert 'Injury_Type' not in conservative
    assert 'Blood_Loss_Severity' not in conservative
    assert 'Recommended_Action' not in baseline


def test_model_comparison_uses_training_pipelines_and_returns_balanced_metrics():
    results = train_compare_models(_sample_records(), 'Injury_Severity', random_state=17)

    assert results['split']['train_records'] + results['split']['test_records'] == 180
    assert results['schema_version'] == MODEL_RESULT_SCHEMA_VERSION
    assert not results['leakage_audit']['target_in_baseline_features']
    assert not results['leakage_audit']['target_in_leakage_aware_features']
    assert results['leakage_audit']['exact_target_copy_columns'] == []
    assert set(results['feature_sets']) == {'baseline', 'leakage_aware'}
    for feature_set in results['feature_sets'].values():
        assert set(feature_set['results']) == {'logistic_regression', 'random_forest'}
        for metrics in feature_set['results'].values():
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['balanced_accuracy'] <= 1
            assert 0 <= metrics['macro_f1'] <= 1
            assert len(metrics['confusion_matrix']) == 3
            assert metrics['feature_count'] > 0
    model = results['feature_sets']['leakage_aware']['fitted_models']['logistic_regression']
    features = results['feature_sets']['leakage_aware']['feature_names']
    prediction = model.predict(_sample_records(1)[features])
    assert len(prediction) == 1


def test_result_schema_drives_the_same_metrics_used_by_comparison_ui():
    results = validate_model_results(train_compare_models(_sample_records(), 'Injury_Severity'), 'Injury_Severity')
    comparison = _comparison_table(results)

    assert len(comparison) == 4
    for row in comparison.to_dict(orient='records'):
        feature_set = 'leakage_aware' if row['Feature set'] == 'Leakage-aware' else 'baseline'
        model_name = row['Model'].lower().replace(' ', '_')
        metrics = results['feature_sets'][feature_set]['results'][model_name]
        assert row['Accuracy'] == metrics['accuracy']
        assert row['Balanced accuracy'] == metrics['balanced_accuracy']
        assert row['Macro F1'] == metrics['macro_f1']
        assert row['Features'] == metrics['feature_count']


def test_model_result_validator_rejects_stale_or_incomplete_contract():
    with pytest.raises(ValueError, match='schema is stale'):
        validate_model_results({'target': 'Injury_Severity', 'results': {}})

    results = train_compare_models(_sample_records(), 'Injury_Severity')
    del results['feature_sets']['leakage_aware']
    with pytest.raises(ValueError, match='exactly the baseline and leakage_aware'):
        validate_model_results(results)


def test_ui_evaluation_rejects_a_stale_cached_result(monkeypatch):
    monkeypatch.setattr(
        model_ui,
        '_cached_model_evaluation',
        lambda *_args: {'target': 'Injury_Severity', 'results': {}},
    )

    with pytest.raises(ValueError, match='schema is stale'):
        model_ui.evaluate_models(_sample_records(), 'Injury_Severity')
