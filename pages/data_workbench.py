from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import COLORS, alert_card, metric_cards, page_header, section_header, style_figure
from config import DEFAULT_DATASET
from src.pipeline import load_and_validate_data, save_uploaded_dataset


def _update_session(result: dict, source_path: str) -> None:
    st.session_state['processed_data'] = result['clean_data']
    st.session_state['quality_report'] = result['quality_report']
    st.session_state['pipeline_summary'] = result['transformation_summary']
    st.session_state['dataset_profile'] = result['profile']
    st.session_state['active_dataset_source'] = source_path
    st.session_state['processed_dataset_path'] = result['processed_path']
    st.session_state['feature_dataset_path'] = result['feature_path']
    st.session_state.pop('dataset_error', None)


def _show_quality(report: dict, row_count: int) -> None:
    metric_cards([
        {'label': 'SOURCE ROWS', 'value': f"{report['row_count']:,}", 'note': f'{row_count:,} rows after duplicate handling', 'accent': COLORS['green']},
        {'label': 'FIELDS', 'value': f"{report['column_count']:,}", 'note': 'source columns', 'accent': COLORS['blue']},
        {'label': 'MISSING CELLS', 'value': f"{report['missing_cells']:,}", 'note': 'not imputed by the pipeline', 'accent': COLORS['amber']},
        {'label': 'DUPLICATE ROWS', 'value': f"{report['duplicate_rows']:,}", 'note': 'exact duplicate rows removed', 'accent': COLORS['red']},
    ])


def _format_bytes(value: int | None) -> str:
    if value is None:
        return 'Not available'
    size = float(value)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.1f} {unit}' if unit != 'B' else f'{int(size)} B'
        size /= 1024
    return f'{size:.1f} GB'


def main():
    page_header(
        'Data Workbench',
        'Inspect the source, review quality, and run a traceable ingestion and transformation pipeline.',
        eyebrow='DATA ENGINEERING',
    )
    with st.expander('Dataset ingestion', expanded=True):
        uploaded_file = st.file_uploader(
            'Choose a CSV or Excel file',
            type=['csv', 'xlsx', 'xls'],
            help='The uploaded source is preserved as-is in data/raw. Cleaned output is written separately to data/processed.',
        )
        if uploaded_file is None:
            current_source = Path(st.session_state.get('active_dataset_source', str(DEFAULT_DATASET)))
            if current_source.exists():
                st.caption(f'Active source: {current_source.name}')
            else:
                st.caption(f'Default project source: {DEFAULT_DATASET.name}')

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(upload_bytes).hexdigest()
        if digest != st.session_state.get('active_upload_digest'):
            try:
                with st.spinner('Validating source schema and running the processing pipeline...'):
                    raw_path = save_uploaded_dataset(upload_bytes, uploaded_file.name)
                    result = load_and_validate_data(raw_path)
                _update_session(result, result['source_path'])
                st.session_state['active_upload_digest'] = digest
                st.session_state['active_source_name'] = uploaded_file.name
                st.success(f'Validated and processed {uploaded_file.name}. The raw upload remains unchanged.')
            except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f'Upload could not be processed: {exc}')
                st.caption('The previous successfully loaded dataset, if any, remains active.')

    df = st.session_state.get('processed_data')
    if df is None or df.empty:
        st.info('No dataset is loaded. Select a supported file above or restore the default project CSV.')
        return

    active_source = st.session_state.get('active_dataset_source', str(DEFAULT_DATASET))
    quality = st.session_state.get('quality_report')
    if quality is None:
        try:
            result = load_and_validate_data(active_source)
            _update_session(result, active_source)
            quality = result['quality_report']
        except (FileNotFoundError, OSError, ValueError) as exc:
            st.error(f'Could not load the source quality report: {exc}')
            return

    profile = st.session_state.get('dataset_profile')
    pipeline = st.session_state.get('pipeline_summary', {})
    section_header(
        'Dataset profile',
        f"Active source: {Path(active_source).name} · "
        f"{_format_bytes(pipeline.get('source_file_size_bytes'))} on disk · "
        f"{profile.get('memory_usage_mb', 0):.2f} MB in memory"
        if profile else f'Active source: {Path(active_source).name}',
    )
    _show_quality(quality, len(df))

    schema_info = quality.get('schema', {})
    schema_col, processing_col = st.columns(2)
    with schema_col:
        section_header('Schema validation', 'Core timestamp and coordinate fields are required for this application.')
        if schema_info.get('valid'):
            st.success('Required schema fields are present; column names are unique.')
        else:
            st.error('The source schema has unresolved issues.')
        st.dataframe(
            pd.DataFrame([
                {'Check': 'Required fields', 'Result': ', '.join(schema_info.get('required_columns', []))},
                {'Check': 'Missing required', 'Result': ', '.join(schema_info.get('missing_required_columns', [])) or 'None'},
                {'Check': 'Duplicate field names', 'Result': ', '.join(schema_info.get('duplicate_column_names', [])) or 'None'},
                {'Check': 'Unrecognized fields', 'Result': ', '.join(schema_info.get('unexpected_columns', [])) or 'None'},
            ]),
            width='stretch',
            hide_index=True,
        )
    with processing_col:
        section_header('Pipeline processing', 'Original upload and normalized output are stored separately.')
        stages = pipeline.get('stage_log', [])
        if stages:
            stage_frame = pd.DataFrame(stages).rename(columns={
                'stage': 'Pipeline stage',
                'status': 'Status',
                'duration_seconds': 'Seconds',
                'detail': 'Execution detail',
            })
            st.dataframe(
                stage_frame[['Pipeline stage', 'Status', 'Seconds', 'Execution detail']],
                width='stretch',
                hide_index=True,
            )
        else:
            st.warning('No completed pipeline execution details are available for this dataset.')
        coercion_counts = pipeline.get('invalid_numeric_values_coerced_to_missing', {})
        st.caption(
            f"{len(stages)} stages completed in {pipeline.get('processing_duration_seconds', 0):.3f}s · "
            f"{pipeline.get('duplicate_rows_removed', 0):,} duplicate rows removed · "
            f"{sum(coercion_counts.values()):,} invalid numeric values converted to missing · "
            f"{pipeline.get('engineered_feature_count', 0)} derived features persisted · "
            f"{pipeline.get('rejected_rows', 0):,} rows rejected · "
            f"{pipeline.get('missing_values_imputed', 0)} missing values imputed."
        )
        st.caption(
            f"Cleaned output: {_format_bytes(pipeline.get('processed_file_size_bytes'))} · "
            f"Engineered output: {_format_bytes(pipeline.get('feature_file_size_bytes'))}. "
            f"{pipeline.get('rejection_policy', 'Rows with quality issues are preserved for review.')}"
        )
        if pipeline.get('normalized_fields'):
            st.caption(f"Normalized fields: {', '.join(pipeline['normalized_fields'])}")

    section_header('Data quality checks', 'Quality counts are calculated against the raw source before normalization.')
    quality_left, quality_right = st.columns(2)
    with quality_left:
        alert_card('Timestamp parsing', f"{quality['invalid_timestamp_count']:,} source timestamps could not be parsed.", 'amber' if quality['invalid_timestamp_count'] else 'green')
        alert_card('Coordinate bounds', f"{quality['invalid_coordinate_count']:,} source rows have missing, non-numeric, or out-of-range coordinates.", 'amber' if quality['invalid_coordinate_count'] else 'green')
        range_issues = quality.get('invalid_range_issues', {})
        if range_issues:
            range_frame = pd.DataFrame([{'Field': field, 'Invalid values': count} for field, count in range_issues.items()])
            st.dataframe(range_frame, width='stretch', hide_index=True)
        else:
            alert_card('Numeric range checks', 'No out-of-range values detected in the configured range-checked fields.', 'green')
    with quality_right:
        missing = pd.Series(quality['missing_values'], dtype='int64').sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            alert_card('Missing-value analysis', 'No missing values were detected in the source dataset.', 'green')
        else:
            missing_frame = missing.rename_axis('Field').rename('Missing cells').reset_index()
            figure = px.bar(
                missing_frame.sort_values('Missing cells'),
                x='Missing cells',
                y='Field',
                orientation='h',
                title='Source missing values',
                color_discrete_sequence=[COLORS['amber']],
            )
            st.plotly_chart(style_figure(figure, max(260, 30 * len(missing_frame))), width='stretch')
            st.dataframe(missing_frame.sort_values('Missing cells', ascending=False), width='stretch', hide_index=True)

    section_header('Schema explorer', 'Search fields and inspect normalized data types.')
    search = st.text_input('Filter schema fields', placeholder='Search field name or data type', label_visibility='collapsed')
    schema = df.dtypes.astype(str).rename('Data type').rename_axis('Field').reset_index()
    if search:
        match = schema['Field'].str.contains(search, case=False, na=False) | schema['Data type'].str.contains(search, case=False, na=False)
        schema = schema[match]
    st.dataframe(schema, width='stretch', hide_index=True, height=320)

    section_header('Data profiling', 'Descriptive statistics and encoded category distributions are computed from processed data.')
    numeric_fields = df.select_dtypes(include='number').columns.tolist()
    profile_col, category_col = st.columns(2)
    if numeric_fields:
        selected_numeric = profile_col.selectbox('Numeric field statistics', numeric_fields)
        values = pd.to_numeric(df[selected_numeric], errors='coerce')
        profile_col.dataframe(
            values.describe().rename_axis('Statistic').rename('Value').to_frame().round(4),
            width='stretch',
        )
    categorical_fields = [
        column for column in df if column in (profile or {}).get('categorical_distributions', {})
    ]
    if categorical_fields:
        selected_categorical = category_col.selectbox('Categorical/code distribution', categorical_fields)
        counts = df[selected_categorical].value_counts(dropna=False).sort_index()
        distribution = pd.DataFrame({
            'Source category': [f'Code {value}' if pd.notna(value) else 'Missing' for value in counts.index],
            'Records': counts.values,
        })
        category_col.dataframe(distribution, width='stretch', hide_index=True)
        category_col.caption('Code meanings are not included in the supplied dataset metadata.')

    section_header('Processed data preview', 'Missing values remain unchanged; timestamp and coordinate normalization are documented above.')
    preview_rows = st.select_slider('Rows to preview', options=[10, 25, 50, 100], value=25)
    st.dataframe(df.head(preview_rows), width='stretch', hide_index=True)

    section_header('Pipeline controls', 'Reprocess the current source or restore the original project dataset.')
    run_col, restore_col = st.columns(2)
    if run_col.button('Run pipeline again', type='primary'):
        try:
            with st.spinner('Revalidating and processing the active source...'):
                refreshed = load_and_validate_data(active_source)
            _update_session(refreshed, active_source)
            st.success(
                f"Pipeline complete · {refreshed['transformation_summary']['source_rows']:,} source rows → "
                f"{refreshed['row_count']:,} processed rows."
            )
            df = refreshed['clean_data']
            quality = refreshed['quality_report']
            profile = refreshed['profile']
            pipeline = refreshed['transformation_summary']
        except (FileNotFoundError, OSError, ValueError) as exc:
            st.error(f'Pipeline execution failed: {exc}')
    if restore_col.button('Reload default project dataset'):
        try:
            with st.spinner('Loading and validating the default project dataset...'):
                refreshed = load_and_validate_data(DEFAULT_DATASET)
            _update_session(refreshed, str(DEFAULT_DATASET))
            st.session_state.pop('active_upload_digest', None)
            st.success('The default project dataset is active again.')
            active_source = str(DEFAULT_DATASET)
            df = refreshed['clean_data']
            quality = refreshed['quality_report']
            profile = refreshed['profile']
            pipeline = refreshed['transformation_summary']
        except (FileNotFoundError, OSError, ValueError) as exc:
            st.error(f'Could not restore the default dataset: {exc}')

    quality_rows = [
        {'check': 'source_rows', 'value': quality['row_count']},
        {'check': 'source_columns', 'value': quality['column_count']},
        {'check': 'duplicate_rows', 'value': quality['duplicate_rows']},
        {'check': 'missing_cells', 'value': quality['missing_cells']},
        {'check': 'invalid_timestamps', 'value': quality['invalid_timestamp_count']},
        {'check': 'invalid_coordinates', 'value': quality['invalid_coordinate_count']},
    ]
    quality_rows.extend(
        {'check': f'invalid_range:{field}', 'value': count}
        for field, count in quality.get('invalid_range_issues', {}).items()
    )
    quality_rows.extend(
        {'check': f'missing:{field}', 'value': count}
        for field, count in quality.get('missing_values', {}).items()
    )
    quality_csv = pd.DataFrame(quality_rows)
    export_cols = st.columns(4)
    export_cols[0].download_button('Download processed CSV', df.to_csv(index=False), 'processed_dataset.csv', 'text/csv', width='stretch')
    export_cols[1].download_button('Download validation report', quality_csv.to_csv(index=False), 'validation_report.csv', 'text/csv', width='stretch')
    export_cols[2].download_button(
        'Download profile summary',
        json.dumps(profile or {}, indent=2, default=str),
        'dataset_profile.json',
        'application/json',
        width='stretch',
    )
    feature_path = Path(st.session_state.get('feature_dataset_path', ''))
    if feature_path.is_file():
        export_cols[3].download_button(
            'Download engineered features',
            feature_path.read_bytes(),
            feature_path.name,
            'text/csv',
            width='stretch',
        )


if __name__ == '__main__':
    main()
