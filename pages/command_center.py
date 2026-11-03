from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import COLORS, metric_cards, page_header, section_header, style_figure
from src.analytics import category_distribution, dataset_summary


def main():
    df = st.session_state.get('processed_data')
    if df is None or df.empty:
        page_header('Command Center', 'A concise summary of the loaded historical observation dataset.')
        st.info('No dataset is loaded. Select a valid source in Data Workbench.')
        return

    page_header(
        'Command Center',
        'A date-filtered overview of historical drone observations, source injury codes, and rescue-priority values.',
        status='HISTORICAL DATA · PROCESSED',
    )
    st.caption('This project dataset is not a live drone telemetry or emergency dispatch feed.')
    filtered = df
    if 'Timestamp' in df:
        timestamps = pd.to_datetime(df['Timestamp'], errors='coerce')
        valid_timestamps = timestamps.dropna()
        if not valid_timestamps.empty:
            date_range = st.date_input(
                'Observation date range',
                value=(valid_timestamps.min().date(), valid_timestamps.max().date()),
                min_value=valid_timestamps.min().date(),
                max_value=valid_timestamps.max().date(),
                key='command_center_date_range',
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                observation_dates = timestamps.dt.date
                filtered = df.loc[observation_dates.between(date_range[0], date_range[1])]
    if filtered.empty:
        st.info('No observations fall within the selected date range.')
        return

    summary = dataset_summary(filtered)
    drone_count = summary['drones']
    disaster_count = summary['disaster_types']
    injury_count = summary['injury_detection_code_1_records']
    metric_cards([
        {'label': 'OBSERVATIONS', 'value': f"{summary['records']:,}", 'note': 'processed dataset records', 'accent': COLORS['green']},
        {'label': 'UNIQUE DRONES', 'value': f'{drone_count:,}' if drone_count is not None else '—', 'note': 'distinct source IDs', 'accent': COLORS['blue']},
        {'label': 'DISASTER CATEGORIES', 'value': str(disaster_count) if disaster_count is not None else '—', 'note': 'distinct encoded source categories', 'accent': COLORS['sage']},
        {'label': 'INJURY DETECTION · CODE 1', 'value': f'{injury_count:,}' if injury_count is not None else '—', 'note': 'records with source value 1', 'accent': COLORS['amber']},
    ])

    section_header('Observation composition', 'Category values are shown with their source codes; the supplied dataset contains no descriptive codebook.')
    distribution_col, severity_col = st.columns(2)
    if 'Disaster_Type' in filtered:
        distribution = category_distribution(filtered, 'Disaster_Type')
        figure = px.bar(
            distribution,
            x='category',
            y='records',
            labels={'category': 'Disaster type', 'records': 'Observations'},
            title='Observations by disaster category',
            color_discrete_sequence=[COLORS['green']],
        )
        distribution_col.plotly_chart(style_figure(figure, 300), width='stretch')
    if 'Injury_Severity' in filtered:
        distribution = category_distribution(filtered, 'Injury_Severity')
        figure = px.bar(
            distribution,
            x='category',
            y='records',
            labels={'category': 'Injury severity', 'records': 'Observations'},
            title='Observations by injury severity',
            color='category',
            color_discrete_sequence=[COLORS['success'], COLORS['amber'], COLORS['red']],
        )
        figure.update_layout(showlegend=False)
        severity_col.plotly_chart(style_figure(figure, 300), width='stretch')

    section_header('Priority and operational status', 'Summaries describe the dataset values, not current field readiness.')
    priority_col, status_col = st.columns(2)
    if 'Rescue_Priority_Score' in filtered:
        priority = pd.to_numeric(filtered['Rescue_Priority_Score'], errors='coerce').dropna()
        if not priority.empty:
            priority_col.metric('Mean rescue-priority score', f'{priority.mean():.3f}')
            priority_col.caption(f'Median {priority.median():.3f} · observed range {priority.min():.3f}–{priority.max():.3f}')
            figure = px.histogram(
                priority,
                nbins=28,
                labels={'value': 'Source priority score', 'count': 'Observations'},
                title='Rescue-priority score distribution',
                color_discrete_sequence=[COLORS['blue']],
            )
            priority_col.plotly_chart(style_figure(figure, 290), width='stretch')
    if 'Drone_Operational_Status' in filtered:
        status_counts = category_distribution(filtered, 'Drone_Operational_Status')
        figure = px.pie(
            status_counts,
            names='category',
            values='records',
            hole=.48,
            title='Operational-status codes',
            color_discrete_sequence=[COLORS['green'], COLORS['sage'], COLORS['amber'], COLORS['red']],
        )
        status_col.plotly_chart(style_figure(figure, 290), width='stretch')
        status_col.caption('The source does not define the meaning of individual status codes.')

    section_header('Observation activity', 'Monthly record counts are calculated from the selected date range and processed timestamps.')
    if 'Timestamp' in filtered:
        activity = filtered.assign(_month=pd.to_datetime(filtered['Timestamp'], errors='coerce').dt.to_period('M'))
        activity = activity.dropna(subset=['_month'])
        monthly = activity.groupby('_month').size().rename('Observations').reset_index()
        monthly['Month'] = monthly['_month'].dt.to_timestamp()
        activity_col, injury_col = st.columns(2)
        figure = px.line(
            monthly,
            x='Month',
            y='Observations',
            markers=True,
            title='Observations per month',
            color_discrete_sequence=[COLORS['green']],
        )
        activity_col.plotly_chart(style_figure(figure, 310), width='stretch')
        if 'Injury_Detection' in activity:
            injury_monthly = (
                activity.assign(_detected=activity['Injury_Detection'].eq(1))
                .groupby('_month')['_detected']
                .sum()
                .rename('Code 1 records')
                .reset_index()
            )
            injury_monthly['Month'] = injury_monthly['_month'].dt.to_timestamp()
            injury_figure = px.line(
                injury_monthly,
                x='Month',
                y='Code 1 records',
                markers=True,
                title='Injury-detection code 1 records per month',
                color_discrete_sequence=[COLORS['red']],
            )
            injury_col.plotly_chart(style_figure(injury_figure, 310), width='stretch')
    else:
        st.info('No valid timestamps are available for the selected records.')

    section_header('Recent observations', 'Most recent timestamps present in the filtered historical dataset.')
    if 'Timestamp' in filtered:
        recent = filtered.assign(_sort_timestamp=pd.to_datetime(filtered['Timestamp'], errors='coerce'))
        recent = recent.sort_values('_sort_timestamp', ascending=False, na_position='last').drop(columns='_sort_timestamp')
    else:
        recent = filtered
    preferred = [
        column for column in [
            'Timestamp', 'Drone_ID', 'Disaster_Type', 'Injury_Detection',
            'Injury_Severity', 'Rescue_Priority_Score', 'Battery_Level_percent',
            'Latitude', 'Longitude',
        ] if column in recent
    ]
    st.dataframe(recent[preferred].head(12), width='stretch', hide_index=True)


if __name__ == '__main__':
    main()
