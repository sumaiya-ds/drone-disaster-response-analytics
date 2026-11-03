from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import chi2_contingency

from src.pipeline import CATEGORICAL_COLUMNS

TARGET_COLUMNS = {'Injury_Severity', 'Injury_Detection'}
MODEL_RESULT_SCHEMA_VERSION = 2
FEATURE_SET_NAMES = {'baseline', 'leakage_aware'}
MODEL_NAMES = {'logistic_regression', 'random_forest'}
BASELINE_EXCLUDED_COLUMNS = {
    'Timestamp',
    'Drone_ID',
    'Recommended_Action',
    'Injury_Detection',
    'Injury_Severity',
    'Rescue_Priority_Score',
    'Victim_Cooperation_Level',
    'Drone_Recharge_Station_Distance_km',
    'Drone_Operational_Status',
}

# These are observations made during injury assessment, rather than pre-assessment
# environmental or drone telemetry. Their availability before the target is unknown.
POST_ASSESSMENT_FEATURES = {
    'Injury_Type',
    'Blood_Loss_Severity',
    'Pain_Level_Score',
    'Respiration_Rate_breaths_per_min',
    'Heart_Rate_bpm',
}
DOWNSTREAM_FEATURES = {
    'Recommended_Action',
    'Rescue_Priority_Score',
    'Injury_Detection',
    'Injury_Severity',
}


def prepare_model_features(
    df: pd.DataFrame,
    target: str = 'Injury_Severity',
    leakage_aware: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target not in df:
        raise ValueError(f'Target column is not present: {target}')
    if target not in TARGET_COLUMNS:
        raise ValueError(f'Unsupported classification target: {target}')

    excluded = set(BASELINE_EXCLUDED_COLUMNS)
    if leakage_aware:
        excluded |= POST_ASSESSMENT_FEATURES | DOWNSTREAM_FEATURES
    excluded.add(target)

    feature_columns = [
        column for column in df.columns
        if column not in excluded
        and (pd.api.types.is_numeric_dtype(df[column]) or column in CATEGORICAL_COLUMNS)
    ]
    if not feature_columns:
        raise ValueError('No eligible feature columns remain after excluding target and leakage-prone fields.')

    features = df[feature_columns].copy()
    categorical = [column for column in feature_columns if column in CATEGORICAL_COLUMNS]
    for column in categorical:
        features[column] = features[column].astype('string')
    return features, df[target].copy(), feature_columns


def _build_preprocessor(features: pd.DataFrame, feature_columns: list[str], model_name: str) -> ColumnTransformer:
    categorical = [column for column in feature_columns if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in feature_columns if column not in categorical]
    transformers = []

    if numeric:
        numeric_steps: list[tuple[str, Any]] = [('imputer', SimpleImputer(strategy='median'))]
        if model_name == 'logistic_regression':
            numeric_steps.append(('scaler', StandardScaler()))
        transformers.append(('numeric', Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])
        transformers.append(('categorical', categorical_pipeline, categorical))
    return ColumnTransformer(transformers, remainder='drop')


def _model(model_name: str):
    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=1000, class_weight='balanced')
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1,
    )


def _evaluate_feature_set(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
    feature_columns: list[str],
) -> dict[str, Any]:
    labels = sorted(train_target.unique().tolist())
    results: dict[str, Any] = {}
    fitted_models = {}

    for model_name in ('logistic_regression', 'random_forest'):
        pipeline = Pipeline([
            ('preprocessor', _build_preprocessor(train_features, feature_columns, model_name)),
            ('model', _model(model_name)),
        ])
        pipeline.fit(train_features, train_target)
        predictions = pipeline.predict(test_features)
        report = classification_report(
            test_target,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        feature_importance = None
        fitted_estimator = pipeline.named_steps['model']
        if hasattr(fitted_estimator, 'feature_importances_'):
            try:
                transformed_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                feature_importance = dict(sorted(
                    zip(transformed_names, fitted_estimator.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )[:15])
            except (AttributeError, ValueError):
                feature_importance = None

        results[model_name] = {
            'accuracy': float(accuracy_score(test_target, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(test_target, predictions)),
            'macro_f1': float(f1_score(test_target, predictions, average='macro', zero_division=0)),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(test_target, predictions, labels=labels).tolist(),
            'labels': labels,
            'feature_importance': feature_importance,
            'feature_count': int(len(feature_columns)),
        }
        fitted_models[model_name] = pipeline
    return {'results': results, 'fitted_models': fitted_models}


def audit_target_relationships(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Describe univariate links for possible proxies without selecting features by test results."""
    candidates = [
        'Injury_Type',
        'Blood_Loss_Severity',
        'Pain_Level_Score',
        'Respiration_Rate_breaths_per_min',
        'Heart_Rate_bpm',
        'Recommended_Action',
        'Rescue_Priority_Score',
        'Injury_Detection',
    ]
    rows = []
    for column in candidates:
        if column not in df or column == target:
            continue
        is_exact_copy = bool(df[column].equals(df[target]))
        if column in CATEGORICAL_COLUMNS:
            contingency = pd.crosstab(df[column], df[target])
            if contingency.empty or min(contingency.shape) < 2:
                association = 0.0
            else:
                chi_square = chi2_contingency(contingency, correction=False)[0]
                denominator = len(df) * (min(contingency.shape) - 1)
                association = float((chi_square / denominator) ** .5) if denominator else 0.0
            statistic_name = "Cramer's V"
        else:
            values = pd.to_numeric(df[column], errors='coerce')
            valid = values.notna() & df[target].notna()
            measured = values[valid]
            grouped = df.loc[valid, target]
            grand_mean = measured.mean() if len(measured) else 0.0
            total_sum_squares = float(((measured - grand_mean) ** 2).sum())
            between_sum_squares = sum(
                len(group) * (group.mean() - grand_mean) ** 2
                for _, group in measured.groupby(grouped)
            ) if len(measured) else 0.0
            association = (between_sum_squares / total_sum_squares) ** .5 if total_sum_squares else 0.0
            statistic_name = 'Eta'
        rows.append({
            'feature': column,
            'association_statistic': statistic_name,
            'association_value': association,
            'exact_target_copy': is_exact_copy,
            'timing_documented': False,
        })
    return pd.DataFrame(rows)


def train_compare_models(
    df: pd.DataFrame,
    target: str = 'Injury_Severity',
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    baseline_features, target_values, baseline_columns = prepare_model_features(df, target, leakage_aware=False)
    aware_features, _, aware_columns = prepare_model_features(df, target, leakage_aware=True)
    if target_values.nunique() < 2:
        raise ValueError('Model training requires at least two target classes.')
    if target_values.isna().any():
        valid = target_values.notna()
        baseline_features = baseline_features.loc[valid]
        aware_features = aware_features.loc[valid]
        target_values = target_values.loc[valid]
    if target_values.value_counts().min() < 2:
        raise ValueError('Each target class needs at least two records for a stratified holdout split.')

    indices = target_values.index
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=target_values,
    )
    train_target = target_values.loc[train_indices]
    test_target = target_values.loc[test_indices]

    baseline = _evaluate_feature_set(
        baseline_features.loc[train_indices],
        baseline_features.loc[test_indices],
        train_target,
        test_target,
        baseline_columns,
    )
    leakage_aware = _evaluate_feature_set(
        aware_features.loc[train_indices],
        aware_features.loc[test_indices],
        train_target,
        test_target,
        aware_columns,
    )
    excluded_for_aware = sorted(set(baseline_columns) - set(aware_columns))
    exact_target_copies = [
        column for column in df.columns
        if column != target and df[column].equals(df[target])
    ]
    return {
        'schema_version': MODEL_RESULT_SCHEMA_VERSION,
        'target': target,
        'test_size': test_size,
        'random_state': random_state,
        'split': {
            'train_records': int(len(train_indices)),
            'test_records': int(len(test_indices)),
            'class_distribution': {str(key): int(value) for key, value in target_values.value_counts().items()},
        },
        'feature_sets': {
            'baseline': {
                **baseline,
                'feature_names': baseline_columns,
                'excluded_features': sorted(BASELINE_EXCLUDED_COLUMNS & set(df.columns)),
            },
            'leakage_aware': {
                **leakage_aware,
                'feature_names': aware_columns,
                'excluded_features': excluded_for_aware,
            },
        },
        'leakage_audit': {
            'target_in_baseline_features': target in baseline_columns,
            'target_in_leakage_aware_features': target in aware_columns,
            'exact_target_copy_columns': exact_target_copies,
            'post_assessment_features_excluded': sorted(POST_ASSESSMENT_FEATURES & set(df.columns)),
            'downstream_features_excluded': sorted(DOWNSTREAM_FEATURES & set(df.columns)),
            'exclusion_rationale': (
                'The conservative feature set models severity from pre-assessment context and telemetry. '
                'Injury type, blood-loss severity, pain, and vital signs are clinical assessments whose '
                'collection timing is undocumented; action recommendations and rescue priority are downstream '
                'decision fields. They are excluded to avoid using information that may only exist after assessment.'
            ),
            'feature_relationships': audit_target_relationships(df, target).to_dict(orient='records'),
        },
    }


def validate_model_results(results: dict[str, Any], target: str | None = None) -> dict[str, Any]:
    """Enforce the model/UI contract and fail explicitly on stale or malformed results."""
    if results.get('schema_version') != MODEL_RESULT_SCHEMA_VERSION:
        raise ValueError(
            'Model evaluation result schema is stale or unsupported. '
            f"Expected version {MODEL_RESULT_SCHEMA_VERSION}, got {results.get('schema_version')!r}."
        )
    if target is not None and results.get('target') != target:
        raise ValueError(
            f"Model evaluation target mismatch: expected {target!r}, got {results.get('target')!r}."
        )
    feature_sets = results.get('feature_sets')
    if not isinstance(feature_sets, dict) or set(feature_sets) != FEATURE_SET_NAMES:
        raise ValueError(
            'Model evaluation must include exactly the baseline and leakage_aware feature sets.'
        )
    if results.get('leakage_audit', {}).get('target_in_baseline_features') or results.get(
        'leakage_audit', {}
    ).get('target_in_leakage_aware_features'):
        raise ValueError('The selected target appears in a model feature set.')

    for feature_set_name, feature_set in feature_sets.items():
        if not isinstance(feature_set, dict):
            raise ValueError(f'Feature set {feature_set_name!r} must be a mapping.')
        feature_names = feature_set.get('feature_names')
        models = feature_set.get('results')
        fitted_models = feature_set.get('fitted_models')
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError(f'Feature set {feature_set_name!r} has no feature names.')
        if results.get('target') in feature_names:
            raise ValueError(f'Target is included in feature set {feature_set_name!r}.')
        if not isinstance(models, dict) or set(models) != MODEL_NAMES:
            raise ValueError(f'Feature set {feature_set_name!r} must evaluate both supported classifiers.')
        if not isinstance(fitted_models, dict) or set(fitted_models) != MODEL_NAMES:
            raise ValueError(f'Feature set {feature_set_name!r} is missing fitted classifiers.')

        for model_name, metrics in models.items():
            if fitted_models[model_name] is None:
                raise ValueError(f'{feature_set_name}/{model_name} has no fitted pipeline.')
            for metric_name in ('accuracy', 'balanced_accuracy', 'macro_f1'):
                if metric_name not in metrics:
                    raise ValueError(f'{feature_set_name}/{model_name} is missing {metric_name}.')
            matrix = metrics.get('confusion_matrix')
            labels = metrics.get('labels')
            report = metrics.get('classification_report')
            if not isinstance(matrix, list) or len(matrix) != len(labels):
                raise ValueError(f'{feature_set_name}/{model_name} has an invalid confusion matrix.')
            if any(not isinstance(row, list) or len(row) != len(labels) for row in matrix):
                raise ValueError(f'{feature_set_name}/{model_name} has an invalid confusion matrix shape.')
            if not isinstance(report, dict):
                raise ValueError(f'{feature_set_name}/{model_name} has no classification report.')
            if int(metrics.get('feature_count', -1)) != len(feature_names):
                raise ValueError(f'{feature_set_name}/{model_name} feature count does not match its feature set.')
    return results
