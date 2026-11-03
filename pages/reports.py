from __future__ import annotations

import pandas as pd
import streamlit as st

from components.ui import COLORS, evaluate_models, metric_cards, page_header, section_header
from config import REPORTS_DIR
from src.analytics import build_summary_table, dataset_summary
from src.exporter import export_reports
from src.reporting import build_executive_report, export_ml_report


def _quality_summary(report: dict, df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ('Source rows', report.get('row_count', len(df)), 'Measured'),
        ('Processed rows', len(df), 'Complete'),
        ('Source fields', report.get('column_count', len(df.columns)), 'Measured'),
        ('Exact duplicates', report.get('duplicate_rows', 0), 'Review' if report.get('duplicate_rows', 0) else 'Clear'),
        ('Missing cells', report.get('missing_cells', 0), 'Review' if report.get('missing_cells', 0) else 'Clear'),
        ('Invalid timestamps', report.get('invalid_timestamp_count', 0), 'Review' if report.get('invalid_timestamp_count', 0) else 'Clear'),
        ('Invalid coordinates', report.get('invalid_coordinate_count', 0), 'Review' if report.get('invalid_coordinate_count', 0) else 'Clear'),
        ('Out-of-range numeric fields', sum(report.get('invalid_range_issues', {}).values()), 'Review' if report.get('invalid_range_issues') else 'Clear'),
    ]
    return pd.DataFrame([
        {'Validation check': name, 'Count': int(value), 'Status': status}
        for name, value, status in checks
    ])


def _model_summary(results: dict) -> pd.DataFrame:
    rows = []
    for feature_set, details in results['feature_sets'].items():
        for model_name, metrics in details['results'].items():
            rows.append({
                'Feature set': 'Leakage-aware' if feature_set == 'leakage_aware' else 'Baseline',
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Balanced accuracy': metrics['balanced_accuracy'],
                'Macro F1': metrics['macro_f1'],
                'Features': metrics['feature_count'],
            })
    return pd.DataFrame(rows)


def main():
    df = st.session_state.get('processed_data')
    page_header(
        'Reports',
        'A concise, traceable summary of the active dataset, processing, quality, geography, and model evaluation.',
        eyebrow='REPORTS & EXPORTS',
    )
    if df is None or df.empty:
        st.info('No dataset is loaded. Select a valid source in Data Workbench.')
        return

    summary = dataset_summary(df)
    quality = st.session_state.get('quality_report') or {}
    pipeline = st.session_state.get('pipeline_summary') or {}
    profile = st.session_state.get('dataset_profile') or {}
    summary_table = build_summary_table(df)
    code_distribution = summary.get('disaster_distribution', {})
    top_code = max(code_distribution, key=code_distribution.get) if code_distribution else None
    top_count = code_distribution.get(top_code, 0) if top_code is not None else 0
    drone_text = f'{summary["drones"]:,}' if summary['drones'] is not None else 'Not available'
    priority_text = f'{summary["avg_priority"]:.3f}' if summary['avg_priority'] is not None else 'Not available'

    section_header('Executive summary', 'Computed from the active dataset; historical records are not a live operational picture.')
    top_line = (
        f'The processed dataset contains {summary["records"]:,} observations from '
        f'{drone_text} distinct drone IDs. '
        f'Disaster source code {top_code} is most frequent ({top_count:,} observations). '
        f'Mean rescue-priority score: {priority_text}.'
        if top_code is not None else f'The processed dataset contains {summary["records"]:,} observations.'
    )
    st.markdown(
        f'<div class="app-panel"><div class="app-panel-title">Dataset briefing</div>'
        f'<div class="app-panel-note">{top_line} These figures describe the supplied data only.</div></div>',
        unsafe_allow_html=True,
    )
    metric_cards([
        {'label': 'PROCESSED RECORDS', 'value': f"{summary['records']:,}", 'note': f"{quality.get('row_count', summary['records']):,} source records", 'accent': COLORS['green']},
        {'label': 'UNIQUE DRONE IDS', 'value': f"{summary['drones']:,}" if summary['drones'] is not None else '—', 'note': 'distinct source identifiers', 'accent': COLORS['blue']},
        {'label': 'DISASTER CATEGORIES', 'value': f"{summary['disaster_types']:,}" if summary['disaster_types'] is not None else '—', 'note': 'distinct numeric source codes', 'accent': COLORS['sage']},
        {'label': 'MEAN PRIORITY SCORE', 'value': f"{summary['avg_priority']:.3f}" if summary['avg_priority'] is not None else '—', 'note': 'source-score mean', 'accent': COLORS['amber']},
    ])

    section_header('Dataset profile', 'Record shape and field type counts from the active processed dataset.')
    metric_cards([
        {'label': 'FIELDS', 'value': f'{len(df.columns):,}', 'note': 'processed schema columns', 'accent': COLORS['green']},
        {'label': 'NUMERIC FIELDS', 'value': f"{len(profile.get('numeric_statistics', {})):,}", 'note': 'profiled numeric fields', 'accent': COLORS['blue']},
        {'label': 'CATEGORY / TEXT FIELDS', 'value': f"{len(profile.get('categorical_distributions', {})):,}", 'note': f"{profile.get('memory_usage_mb', 0):.2f} MB memory footprint", 'accent': COLORS['sage']},
    ])

    section_header('Pipeline and data quality', 'Ingestion, schema validation, normalization, and duplicate handling summary.')
    quality_col, pipeline_col = st.columns(2)
    quality_table = _quality_summary(quality, df)
    quality_col.dataframe(quality_table, width='stretch', hide_index=True)
    pipeline_table = pd.DataFrame([
        {'Pipeline step': 'Source rows', 'Result': f"{pipeline.get('source_rows', quality.get('row_count', len(df))):,}"},
        {'Pipeline step': 'Duplicate rows removed', 'Result': f"{pipeline.get('duplicate_rows_removed', 0):,}"},
        {'Pipeline step': 'Processed rows', 'Result': f"{pipeline.get('processed_rows', len(df)):,}"},
        {'Pipeline step': 'Derived feature fields', 'Result': f"{pipeline.get('engineered_feature_count', 0):,}"},
        {'Pipeline step': 'Normalized fields', 'Result': ', '.join(pipeline.get('normalized_fields', [])) or 'None'},
        {'Pipeline step': 'Missing values imputed', 'Result': str(pipeline.get('missing_values_imputed', 0))},
        {'Pipeline step': 'Active source', 'Result': st.session_state.get('active_dataset_source', 'Project dataset')},
        {'Pipeline step': 'Processed storage', 'Result': st.session_state.get('processed_dataset_path', 'Not recorded')},
        {'Pipeline step': 'Feature storage', 'Result': st.session_state.get('feature_dataset_path', 'Not recorded')},
    ])
    pipeline_col.dataframe(pipeline_table, width='stretch', hide_index=True)
    if quality.get('missing_cells', 0):
        st.caption('Missing values are preserved rather than imputed by the data pipeline; inspect the workbench quality report.')
    else:
        st.success('The source quality report contains no missing cells.')

    section_header('Disaster and injury analytics', 'Source category values are retained; no unsupported code meanings are assigned.')
    analytics_col, geo_col = st.columns(2)
    analytics_col.dataframe(summary_table, width='stretch', hide_index=True)
    if {'Latitude', 'Longitude'}.issubset(df.columns):
        latitude = pd.to_numeric(df['Latitude'], errors='coerce')
        longitude = pd.to_numeric(df['Longitude'], errors='coerce')
        valid = (
            latitude.notna() & longitude.notna()
            & latitude.between(-90, 90) & longitude.between(-180, 180)
        )
        geo_table = pd.DataFrame([
            {'Geographic measure': 'Valid coordinate pairs', 'Value': f'{int(valid.sum()):,}'},
            {'Geographic measure': 'Excluded / invalid', 'Value': f'{int((~valid).sum()):,}'},
            {'Geographic measure': 'Latitude coverage', 'Value': f'{latitude[valid].min():.4f} to {latitude[valid].max():.4f}' if valid.any() else 'Not available'},
            {'Geographic measure': 'Longitude coverage', 'Value': f'{longitude[valid].min():.4f} to {longitude[valid].max():.4f}' if valid.any() else 'Not available'},
        ])
        geo_col.dataframe(geo_table, width='stretch', hide_index=True)
        geo_col.caption('Coordinates are not independently verified as incident locations.')

    section_header('Model evaluation summary', 'Target, feature-set definitions, and metrics are computed using a stratified holdout split.')
    target_options = [column for column in ('Injury_Severity', 'Injury_Detection') if column in df]
    model_target = st.selectbox('Model target', target_options, key='report_target') if target_options else None
    model_results = None
    if model_target:
        try:
            model_results = evaluate_models(df, model_target)
            st.session_state['model_evaluation'] = model_results
            model_frame = _model_summary(model_results)
            st.dataframe(model_frame.style.format({
                'Accuracy': '{:.3f}',
                'Balanced accuracy': '{:.3f}',
                'Macro F1': '{:.3f}',
            }), width='stretch', hide_index=True)
        except (ValueError, KeyError) as exc:
            st.warning(f'Model evaluation could not be generated: {exc}')

    section_header('Downloadable outputs', 'Download reports for presentation or continue to generate local files.')
    downloads = st.columns(3)
    downloads[0].download_button(
        'Analytics summary CSV',
        summary_table.to_csv(index=False),
        'analytics_summary.csv',
        'text/csv',
        width='stretch',
    )
    downloads[1].download_button(
        'Quality report CSV',
        quality_table.to_csv(index=False),
        'data_quality_report.csv',
        'text/csv',
        width='stretch',
    )
    executive_report = build_executive_report(summary, quality, pipeline, model_results)
    downloads[2].download_button(
        'Executive summary TXT',
        executive_report,
        'executive_summary.txt',
        'text/plain',
        width='stretch',
    )
    with st.expander('Export local report files'):
        if st.button('Generate report files', type='primary'):
            try:
                files = export_reports(summary_table, df, REPORTS_DIR)
                quality_path = REPORTS_DIR / 'data_quality_report.csv'
                quality_table.to_csv(quality_path, index=False)
                files.append(str(quality_path))
                (REPORTS_DIR / 'executive_summary.txt').write_text(executive_report, encoding='utf-8')
                files.append(str(REPORTS_DIR / 'executive_summary.txt'))
                if model_results:
                    ml_path = REPORTS_DIR / 'ml_evaluation_report.csv'
                    export_ml_report(model_results, ml_path)
                    files.append(str(ml_path))
                st.success(f'Generated {len(files)} report files.')
                for path in files:
                    st.caption(path)
            except OSError as exc:
                st.error(f'Report export failed: {exc}')


if __name__ == '__main__':
    main()
