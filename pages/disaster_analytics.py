from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import COLORS, metric_cards, page_header, section_header, style_figure
from src.analytics import build_disaster_charts, category_distribution


def _coded_chart(df: pd.DataFrame, column: str, title: str, color: str):
    distribution = category_distribution(df, column)
    if distribution.empty:
        return None
    figure = px.bar(
        distribution,
        x='category',
        y='records',
        labels={'category': 'Source category', 'records': 'Observations'},
        title=title,
        color_discrete_sequence=[color],
    )
    return figure


def _source_code(value) -> str:
    return 'Missing' if pd.isna(value) else f'Code {value}'


def main():
    df = st.session_state.get('processed_data')
    page_header(
        'Disaster Analytics',
        'Filter and compare disaster observations, injury codes, environmental readings, and operational patterns.',
        eyebrow='ANALYTICAL WORKSPACE',
        status='HISTORICAL DATA',
    )
    if df is None or df.empty:
        st.info('No dataset is loaded. Select a valid source in Data Workbench.')
        return

    filter_cols = st.columns(4)
    filtered = df
    for column, slot in zip(
        ('Disaster_Type', 'Injury_Severity', 'Injury_Detection', 'Drone_Operational_Status'),
        filter_cols,
    ):
        if column in df:
            values = sorted(df[column].dropna().unique().tolist())
            selected = slot.multiselect(
                column.replace('_', ' '),
                values,
                default=values,
                format_func=lambda value: f'Code {value}',
                key=f'analytics_filter_{column}',
            )
            filtered = filtered[filtered[column].isin(selected)]

    date_range = None
    if 'Timestamp' in filtered:
        timestamps = pd.to_datetime(filtered['Timestamp'], errors='coerce').dropna()
        if not timestamps.empty:
            date_range = st.date_input(
                'Observation date range',
                value=(timestamps.min().date(), timestamps.max().date()),
                min_value=timestamps.min().date(),
                max_value=timestamps.max().date(),
                key='analytics_date_range',
            )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        observation_dates = pd.to_datetime(filtered['Timestamp'], errors='coerce').dt.date
        filtered = filtered[observation_dates.between(date_range[0], date_range[1])]
    if filtered.empty:
        st.info('No observations match the selected filters.')
        return

    injury_count = int((filtered['Injury_Detection'] == 1).sum()) if 'Injury_Detection' in filtered else 0
    priority = pd.to_numeric(filtered['Rescue_Priority_Score'], errors='coerce') if 'Rescue_Priority_Score' in filtered else pd.Series(dtype=float)
    metric_cards([
        {'label': 'OBSERVATIONS', 'value': f'{len(filtered):,}', 'note': f'of {len(df):,} source records', 'accent': COLORS['green']},
        {'label': 'INJURY DETECTION · CODE 1', 'value': f'{injury_count:,}', 'note': 'source-code count, not a decoded label', 'accent': COLORS['red']},
        {'label': 'DISASTER CATEGORIES', 'value': str(filtered['Disaster_Type'].nunique()) if 'Disaster_Type' in filtered else '—', 'note': 'distinct source codes', 'accent': COLORS['blue']},
        {'label': 'MEAN PRIORITY SCORE', 'value': f'{priority.mean():.3f}' if not priority.empty else '—', 'note': 'filtered source score', 'accent': COLORS['amber']},
    ])

    section_header('Disaster and injury distributions', 'Category codes are preserved because no source codebook was supplied.')
    charts = build_disaster_charts(filtered)
    distribution_col, severity_col = st.columns(2)
    if 'disaster_type' in charts:
        distribution_col.plotly_chart(style_figure(charts['disaster_type'], 325), width='stretch')
    if 'severity' in charts:
        severity_col.plotly_chart(style_figure(charts['severity'], 325), width='stretch')

    detection_col, priority_col = st.columns(2)
    if 'Injury_Detection' in filtered:
        figure = _coded_chart(filtered, 'Injury_Detection', 'Injury-detection source codes', COLORS['green'])
        detection_col.plotly_chart(style_figure(figure, 300), width='stretch')
    if not priority.empty:
        figure = px.histogram(
            priority.dropna(),
            nbins=30,
            labels={'value': 'Rescue-priority score', 'count': 'Observations'},
            title='Rescue-priority score distribution',
            color_discrete_sequence=[COLORS['blue']],
        )
        priority_col.plotly_chart(style_figure(figure, 300), width='stretch')

    section_header('Temporal analysis', 'Monthly observation counts derived from timestamps in the filtered records.')
    if 'timeline' in charts:
        charts['timeline'].update_traces(line={'color': COLORS['green'], 'width': 2.2})
        st.plotly_chart(style_figure(charts['timeline'], 330), width='stretch')
    else:
        st.info('No valid timestamps are available for the selected records.')

    if {'Disaster_Type', 'Injury_Severity'}.issubset(filtered.columns):
        section_header(
            'Cross-category comparison',
            'Counts compare source injury-severity codes within each disaster-type code after all filters are applied.',
        )
        comparison = (
            filtered.groupby(['Disaster_Type', 'Injury_Severity'], dropna=False)
            .size()
            .rename('Observations')
            .reset_index()
        )
        comparison['Disaster category'] = comparison['Disaster_Type'].map(_source_code)
        comparison['Injury severity'] = comparison['Injury_Severity'].map(_source_code)
        figure = px.bar(
            comparison,
            x='Disaster category',
            y='Observations',
            color='Injury severity',
            barmode='group',
            title='Injury-severity codes by disaster category',
            color_discrete_sequence=[COLORS['green'], COLORS['amber'], COLORS['red'], COLORS['blue']],
        )
        st.plotly_chart(style_figure(figure, 340), width='stretch')

    section_header('Environmental and operational conditions', 'Encoded environmental and operational fields are identified as source codes.')
    condition_fields = [
        column for column in [
            'Weather_Condition', 'Lighting_Condition', 'Operation_Time_of_Day',
            'Drone_Operational_Status',
        ] if column in filtered
    ]
    if condition_fields:
        chart_cols = st.columns(min(2, len(condition_fields)))
        for index, column in enumerate(condition_fields):
            slot = chart_cols[index % len(chart_cols)]
            color = COLORS['amber'] if column == 'Drone_Operational_Status' else COLORS['sage']
            figure = _coded_chart(filtered, column, column.replace('_', ' '), color)
            slot.plotly_chart(style_figure(figure, 280), width='stretch')
    else:
        st.info('No environmental or operational category fields are available.')


if __name__ == '__main__':
    main()
