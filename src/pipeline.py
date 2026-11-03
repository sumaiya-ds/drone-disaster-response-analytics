from __future__ import annotations

import logging
from time import perf_counter
from pathlib import Path
from typing import Any

import pandas as pd

from config import DEFAULT_CLEANED_DATA, DEFAULT_DATASET, DEFAULT_FEATURE_DATA, PROCESSED_DATA_DIR, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ('Timestamp', 'Latitude', 'Longitude')
EXPECTED_COLUMNS = set(
    REQUIRED_COLUMNS
    + (
        'Drone_ID', 'Altitude_meters', 'Speed_kmph', 'Battery_Level_percent',
        'Signal_Strength_percent', 'Obstacle_Distance_meters', 'Weather_Condition',
        'Temperature_C', 'Humidity_percent', 'Wind_Speed_kmph', 'Lighting_Condition',
        'Disaster_Type', 'Detected_Object_Type', 'Number_of_Individuals_Detected',
        'Injury_Type', 'Proximity_to_Hazard_meters', 'Medical_Supply_Inventory',
        'Nearest_Medical_Facility_km', 'Crowd_Density', 'Area_Accessibility',
        'Energy_Consumption_kWh', 'Drone_Operational_Status', 'Flight_Duration_minutes',
        'Visibility_km', 'Fire_Hazard_Proximity_meters', 'Flood_Level_meters',
        'Debris_Obstacle_Presence', 'Blood_Loss_Severity',
        'Respiration_Rate_breaths_per_min', 'Heart_Rate_bpm', 'Pain_Level_Score',
        'Structural_Collapse_Presence', 'Chemical_Spill_Presence',
        'Hazardous_Area_Class', 'Rescue_Team_Proximity_km',
        'Nearby_Resources_Count', 'Victim_Cooperation_Level',
        'Drone_Recharge_Station_Distance_km', 'Operation_Time_of_Day',
        'Injury_Detection', 'Injury_Severity', 'Recommended_Action',
        'Rescue_Priority_Score',
    )
)
CATEGORICAL_COLUMNS = (
    'Drone_ID', 'Weather_Condition', 'Lighting_Condition', 'Disaster_Type',
    'Detected_Object_Type', 'Injury_Type', 'Blood_Loss_Severity', 'Area_Accessibility',
    'Drone_Operational_Status', 'Debris_Obstacle_Presence',
    'Structural_Collapse_Presence', 'Chemical_Spill_Presence',
    'Hazardous_Area_Class', 'Victim_Cooperation_Level', 'Operation_Time_of_Day',
    'Injury_Detection', 'Injury_Severity', 'Recommended_Action',
)
RANGE_LIMITS = {
    'Battery_Level_percent': (0, 100),
    'Signal_Strength_percent': (0, 100),
    'Humidity_percent': (0, 100),
    'Rescue_Priority_Score': (0, 1),
}
NON_NEGATIVE_COLUMNS = (
    'Altitude_meters', 'Speed_kmph', 'Obstacle_Distance_meters', 'Wind_Speed_kmph',
    'Number_of_Individuals_Detected', 'Proximity_to_Hazard_meters',
    'Medical_Supply_Inventory', 'Nearest_Medical_Facility_km', 'Crowd_Density',
    'Energy_Consumption_kWh', 'Flight_Duration_minutes', 'Visibility_km',
    'Fire_Hazard_Proximity_meters', 'Flood_Level_meters',
    'Respiration_Rate_breaths_per_min', 'Heart_Rate_bpm', 'Pain_Level_Score',
    'Rescue_Team_Proximity_km', 'Nearby_Resources_Count',
    'Drone_Recharge_Station_Distance_km',
)
NUMERIC_COLUMNS = tuple(sorted(EXPECTED_COLUMNS - set(CATEGORICAL_COLUMNS) - {'Timestamp'}))


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f'Unsupported file format: {path.suffix}')


def save_uploaded_dataset(content: bytes, filename: str, raw_dir: str | Path = RAW_DATA_DIR) -> Path:
    """Persist the original upload unchanged under a safe basename."""
    safe_name = Path(filename).name
    if not safe_name or safe_name in {'.', '..'}:
        raise ValueError('Uploaded filename must include a file name.')
    if Path(safe_name).suffix.lower() not in {'.csv', '.xlsx', '.xls'}:
        raise ValueError(f'Unsupported file format: {Path(safe_name).suffix}')
    output_dir = Path(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / safe_name
    destination.write_bytes(content)
    return destination


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_column_names = df.columns[df.columns.duplicated()].astype(str).tolist()
    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    report: dict[str, Any] = {
        'row_count': int(len(df)),
        'column_count': int(len(df.columns)),
        'duplicate_rows': int(df.duplicated().sum()),
        'missing_values': df.isna().sum().to_dict(),
        'missing_cells': int(df.isna().sum().sum()),
        'invalid_timestamp_count': 0,
        'invalid_coordinate_count': 0,
        'invalid_range_issues': {},
        'schema': {
            'valid': not missing_required and not duplicate_column_names,
            'required_columns': list(REQUIRED_COLUMNS),
            'missing_required_columns': missing_required,
            'duplicate_column_names': duplicate_column_names,
            'unexpected_columns': [str(column) for column in df.columns if column not in EXPECTED_COLUMNS],
        },
    }

    if 'Timestamp' in df.columns:
        timestamps = pd.to_datetime(df['Timestamp'], errors='coerce')
        report['invalid_timestamp_count'] = int(timestamps.isna().sum())

    if {'Latitude', 'Longitude'}.issubset(df.columns):
        latitude = pd.to_numeric(df['Latitude'], errors='coerce')
        longitude = pd.to_numeric(df['Longitude'], errors='coerce')
        invalid = (
            latitude.isna() | longitude.isna()
            | latitude.lt(-90) | latitude.gt(90)
            | longitude.lt(-180) | longitude.gt(180)
        )
        report['invalid_coordinate_count'] = int(invalid.sum())

    range_limits = dict(RANGE_LIMITS)
    range_limits.update({column: (0, None) for column in NON_NEGATIVE_COLUMNS})
    for column, (lower, upper) in range_limits.items():
        if column not in df:
            continue
        original = df[column]
        values = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & values.isna()
        if lower is not None:
            invalid |= values.notna() & values.lt(lower)
        if upper is not None:
            invalid |= values.notna() & values.gt(upper)
        count = int(invalid.sum())
        if count:
            report['invalid_range_issues'][column] = count
    return report


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize required data types, remove exact duplicates, and preserve missing/invalid values."""
    input_rows = len(df)
    cleaned = df.drop_duplicates().copy()
    duplicate_rows_removed = input_rows - len(cleaned)
    timestamp_invalid_before = 0
    normalized_fields: list[str] = []

    if 'Timestamp' in cleaned:
        original = cleaned['Timestamp']
        parsed = pd.to_datetime(original, errors='coerce')
        timestamp_invalid_before = int(parsed.isna().sum())
        cleaned['Timestamp'] = parsed
        normalized_fields.append('Timestamp')

    coercion_counts: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        if column in cleaned:
            original = cleaned[column]
            numeric = pd.to_numeric(original, errors='coerce')
            invalid_coercions = original.notna() & numeric.isna()
            if invalid_coercions.any():
                coercion_counts[column] = int(invalid_coercions.sum())
            cleaned[column] = numeric
            normalized_fields.append(column)

    for column in ('Latitude', 'Longitude'):
        if column in cleaned:
            cleaned[column] = pd.to_numeric(cleaned[column], errors='coerce')

    summary = {
        'source_rows': int(input_rows),
        'duplicate_rows_removed': int(duplicate_rows_removed),
        'processed_rows': int(len(cleaned)),
        'normalized_fields': normalized_fields,
        'invalid_numeric_values_coerced_to_missing': coercion_counts,
        'invalid_timestamps_preserved_as_missing': timestamp_invalid_before,
        'missing_values_imputed': 0,
        'coordinates_filled_or_rewritten': 0,
    }
    return cleaned, summary


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = [
        column for column in df.columns
        if column in CATEGORICAL_COLUMNS or not pd.api.types.is_numeric_dtype(df[column])
    ]
    numeric_stats = {}
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors='coerce')
        numeric_stats[column] = {
            'count': int(series.count()),
            'mean': None if series.dropna().empty else float(series.mean()),
            'std': None if series.dropna().empty else float(series.std()),
            'min': None if series.dropna().empty else float(series.min()),
            'median': None if series.dropna().empty else float(series.median()),
            'max': None if series.dropna().empty else float(series.max()),
        }
    distributions = {
        column: {str(value): int(count) for value, count in df[column].value_counts(dropna=False).head(20).items()}
        for column in categorical_columns
    }
    return {
        'row_count': int(len(df)),
        'column_count': int(len(df.columns)),
        'memory_usage_mb': round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 3),
        'numeric_statistics': numeric_stats,
        'categorical_distributions': distributions,
        'dtypes': {str(column): str(dtype) for column, dtype in df.dtypes.items()},
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    if 'Timestamp' in features:
        timestamp = pd.to_datetime(features['Timestamp'], errors='coerce')
        features['hour'] = timestamp.dt.hour
        features['day'] = timestamp.dt.day
        features['month'] = timestamp.dt.month
        hour = timestamp.dt.hour
        is_night = hour.ge(19) | hour.le(5)
        features['is_night'] = is_night.where(hour.notna()).astype('Int64')
    if 'Battery_Level_percent' in features:
        battery = pd.to_numeric(features['Battery_Level_percent'], errors='coerce')
        features['battery_risk_flag'] = battery.lt(20).astype('Int64')
    if {'Latitude', 'Longitude', 'Proximity_to_Hazard_meters'}.issubset(features.columns):
        proximity = pd.to_numeric(features['Proximity_to_Hazard_meters'], errors='coerce')
        latitude = pd.to_numeric(features['Latitude'], errors='coerce')
        longitude = pd.to_numeric(features['Longitude'], errors='coerce')
        features['hazard_risk_score'] = proximity / (latitude.abs() + longitude.abs() + 1)
    if {'Medical_Supply_Inventory', 'Nearby_Resources_Count'}.issubset(features.columns):
        inventory = pd.to_numeric(features['Medical_Supply_Inventory'], errors='coerce')
        resources = pd.to_numeric(features['Nearby_Resources_Count'], errors='coerce')
        features['resource_pressure'] = inventory / (resources + 1)
    if 'Weather_Condition' in features.columns and not pd.api.types.is_numeric_dtype(features['Weather_Condition']):
        features['weather_risk'] = features['Weather_Condition'].astype(str).str.contains(
            'storm|rain|fog|wind', case=False, na=False
        ).astype('Int64')
    if 'Visibility_km' in features:
        features['visibility_risk'] = pd.to_numeric(features['Visibility_km'], errors='coerce').lt(5).astype('Int64')
    return features


def save_processed_data(df: pd.DataFrame, output_path: str | Path = DEFAULT_CLEANED_DATA) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def load_and_validate_data(
    file_path: str | Path = DEFAULT_DATASET,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(file_path)
    pipeline_started = perf_counter()
    stage_log: list[dict[str, Any]] = []

    def record_stage(stage: str, started_at: float, detail: str) -> None:
        elapsed = round(perf_counter() - started_at, 4)
        stage_log.append({
            'stage': stage,
            'status': 'Complete',
            'duration_seconds': elapsed,
            'detail': detail,
        })
        LOGGER.info('Pipeline stage complete: %s in %.4f seconds | %s', stage, elapsed, detail)

    stage_started = perf_counter()
    raw_df = load_dataset(source)
    record_stage('Ingest source', stage_started, f'{len(raw_df):,} source rows loaded')
    if raw_df.empty:
        raise ValueError('The selected dataset contains no rows.')

    stage_started = perf_counter()
    quality_report = validate_dataframe(raw_df)
    record_stage('Validate schema and quality', stage_started, 'Required fields and source quality checks evaluated')
    if quality_report['schema']['duplicate_column_names']:
        raise ValueError(f"Dataset has duplicate column names: {quality_report['schema']['duplicate_column_names']}")
    if quality_report['schema']['missing_required_columns']:
        raise ValueError(
            'Dataset is missing required columns: '
            + ', '.join(quality_report['schema']['missing_required_columns'])
        )

    stage_started = perf_counter()
    cleaned, transformation_summary = clean_dataframe(raw_df)
    record_stage(
        'Normalize values and remove exact duplicates',
        stage_started,
        f"{transformation_summary['duplicate_rows_removed']:,} duplicate rows removed; other source rows preserved",
    )

    stage_started = perf_counter()
    profile = profile_dataframe(cleaned)
    record_stage('Profile processed dataset', stage_started, f'{len(cleaned.columns):,} fields profiled')
    destination = Path(output_path) if output_path else (
        DEFAULT_CLEANED_DATA if source.name == DEFAULT_DATASET.name
        else PROCESSED_DATA_DIR / f'{source.stem}_processed.csv'
    )
    stage_started = perf_counter()
    processed_path = save_processed_data(cleaned, destination)
    record_stage('Persist cleaned dataset', stage_started, f'Saved {processed_path.name}')
    stage_started = perf_counter()
    feature_data = build_feature_matrix(cleaned)
    record_stage(
        'Engineer derived features',
        stage_started,
        f'{len(feature_data.columns) - len(cleaned.columns):,} derived features created',
    )
    if source.name == DEFAULT_DATASET.name:
        feature_path = DEFAULT_FEATURE_DATA
    elif output_path is not None:
        feature_path = destination.with_name(f'{source.stem}_engineered_features.csv')
    else:
        feature_path = PROCESSED_DATA_DIR / f'{source.stem}_engineered_features.csv'
    stage_started = perf_counter()
    save_processed_data(feature_data, feature_path)
    record_stage('Persist engineered dataset', stage_started, f'Saved {Path(feature_path).name}')
    engineered_columns = [column for column in feature_data if column not in cleaned.columns]
    transformation_summary['engineered_feature_count'] = len(engineered_columns)
    transformation_summary['engineered_features'] = engineered_columns
    transformation_summary['stage_log'] = stage_log
    transformation_summary['processing_duration_seconds'] = round(perf_counter() - pipeline_started, 4)
    transformation_summary['rejected_rows'] = 0
    transformation_summary['rejection_policy'] = (
        'No rows rejected; exact duplicates were removed and invalid values were preserved or coerced to missing.'
    )
    transformation_summary['source_file_size_bytes'] = source.stat().st_size
    transformation_summary['processed_file_size_bytes'] = processed_path.stat().st_size
    transformation_summary['feature_file_size_bytes'] = feature_path.stat().st_size
    LOGGER.info(
        'Processed dataset %s: %s source rows, %s processed rows, %s engineered features in %.4f seconds',
        source,
        len(raw_df),
        len(cleaned),
        len(engineered_columns),
        transformation_summary['processing_duration_seconds'],
    )
    return {
        'source_path': str(source),
        'row_count': int(len(cleaned)),
        'quality_report': quality_report,
        'transformation_summary': transformation_summary,
        'profile': profile,
        'clean_data': cleaned,
        'processed_path': str(processed_path),
        'feature_path': str(feature_path),
    }
