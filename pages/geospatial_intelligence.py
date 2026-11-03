from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import COLORS, metric_cards, page_header, section_header, style_figure

MAX_MAP_POINTS = 5000


def main():
    df = st.session_state.get('processed_data')
    page_header(
        'Geospatial Intelligence',
        'Explore historical observation coordinates with category filters and explicit coordinate validation.',
        eyebrow='GEOSPATIAL ANALYSIS',
        status='HISTORICAL COORDINATES',
    )
    if df is None or df.empty:
        st.info('No dataset is loaded. Select a valid source in Data Workbench.')
        return
    if not {'Latitude', 'Longitude'}.issubset(df.columns):
        st.warning('Both Latitude and Longitude fields are required for this view.')
        return

    geo = df.copy()
    geo['Latitude'] = pd.to_numeric(geo['Latitude'], errors='coerce')
    geo['Longitude'] = pd.to_numeric(geo['Longitude'], errors='coerce')
    invalid_coordinates = (
        geo['Latitude'].isna() | geo['Longitude'].isna()
        | geo['Latitude'].lt(-90) | geo['Latitude'].gt(90)
        | geo['Longitude'].lt(-180) | geo['Longitude'].gt(180)
    )
    invalid_count = int(invalid_coordinates.sum())
    valid_geo = geo.loc[~invalid_coordinates].copy()
    if valid_geo.empty:
        st.warning('No valid coordinates are available for mapping.')
        return

    filters = st.columns(4)
    if 'Disaster_Type' in valid_geo:
        values = sorted(valid_geo['Disaster_Type'].dropna().unique().tolist())
        selected = filters[0].multiselect('Disaster category', values, default=values, format_func=lambda value: f'Code {value}', key='geo_disaster')
        valid_geo = valid_geo[valid_geo['Disaster_Type'].isin(selected)]
    if 'Injury_Severity' in valid_geo:
        values = sorted(valid_geo['Injury_Severity'].dropna().unique().tolist())
        selected = filters[1].multiselect('Injury severity', values, default=values, format_func=lambda value: f'Code {value}', key='geo_severity')
        valid_geo = valid_geo[valid_geo['Injury_Severity'].isin(selected)]
    if 'Drone_Operational_Status' in valid_geo:
        values = sorted(valid_geo['Drone_Operational_Status'].dropna().unique().tolist())
        selected = filters[2].multiselect(
            'Operational status',
            values,
            default=values,
            format_func=lambda value: f'Code {value}',
            key='geo_operational_status',
        )
        valid_geo = valid_geo[valid_geo['Drone_Operational_Status'].isin(selected)]
    if 'Rescue_Priority_Score' in valid_geo:
        valid_geo['Rescue_Priority_Score'] = pd.to_numeric(valid_geo['Rescue_Priority_Score'], errors='coerce')
        scores = valid_geo['Rescue_Priority_Score'].dropna()
        if not scores.empty:
            low, high = float(scores.min()), float(scores.max())
            score_range = filters[3].slider(
                'Rescue-priority range',
                min_value=low,
                max_value=high,
                value=(low, high),
                step=(high - low) / 100 if high > low else 1.0,
                format='%.3f',
            )
            valid_geo = valid_geo[valid_geo['Rescue_Priority_Score'].between(*score_range, inclusive='both')]

    if 'Timestamp' in valid_geo:
        timestamps = pd.to_datetime(valid_geo['Timestamp'], errors='coerce')
        valid_timestamps = timestamps.dropna()
        if not valid_timestamps.empty:
            date_range = st.date_input(
                'Observation date range',
                value=(valid_timestamps.min().date(), valid_timestamps.max().date()),
                min_value=valid_timestamps.min().date(),
                max_value=valid_timestamps.max().date(),
                key='geo_date_range',
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                valid_geo = valid_geo.loc[
                    timestamps.dt.date.between(date_range[0], date_range[1])
                ]

    if valid_geo.empty:
        st.info('No observations match the selected filters.')
        return

    score_mean = valid_geo['Rescue_Priority_Score'].mean() if 'Rescue_Priority_Score' in valid_geo else None
    metric_cards([
        {'label': 'VALID LOCATIONS', 'value': f'{len(valid_geo):,}', 'note': f'{invalid_count:,} invalid of {len(df):,} source records', 'accent': COLORS['green']},
        {'label': 'LATITUDE RANGE', 'value': f'{valid_geo["Latitude"].min():.3f}°–{valid_geo["Latitude"].max():.3f}°', 'note': 'filtered valid observations', 'accent': COLORS['blue']},
        {'label': 'LONGITUDE RANGE', 'value': f'{valid_geo["Longitude"].min():.3f}°–{valid_geo["Longitude"].max():.3f}°', 'note': 'filtered valid observations', 'accent': COLORS['sage']},
        {'label': 'MEAN PRIORITY', 'value': f'{score_mean:.3f}' if score_mean is not None else 'Not available', 'note': 'filtered source score', 'accent': COLORS['amber']},
    ])

    section_header(
        'Observation map',
        f'Up to {MAX_MAP_POINTS:,} observations are displayed for responsiveness. Coordinates are source values, not verified incident locations.',
    )
    plot_data = valid_geo
    sampled = len(valid_geo) > MAX_MAP_POINTS
    if sampled:
        plot_data = valid_geo.sample(MAX_MAP_POINTS, random_state=42)
    severity_column = None
    color_map = None
    if 'Injury_Severity' in plot_data:
        severity_column = 'Injury severity'
        plot_data = plot_data.copy()
        plot_data[severity_column] = plot_data['Injury_Severity'].map(lambda value: f'Code {value}')
        palette = [COLORS['success'], COLORS['amber'], COLORS['red'], COLORS['blue'], '#8D7866']
        codes = sorted(plot_data['Injury_Severity'].dropna().unique())
        color_map = {f'Code {code}': palette[index % len(palette)] for index, code in enumerate(codes)}

    hover = {
        column: True for column in [
            'Timestamp', 'Drone_ID', 'Disaster_Type', 'Injury_Severity',
            'Rescue_Priority_Score',
        ] if column in plot_data
    }
    figure = px.scatter_geo(
        plot_data,
        lat='Latitude',
        lon='Longitude',
        color=severity_column,
        color_discrete_map=color_map,
        hover_name='Drone_ID' if 'Drone_ID' in plot_data else None,
        hover_data=hover,
        projection='mercator',
        title='Historical drone observation coordinates',
    )
    lat_min, lat_max = float(valid_geo['Latitude'].min()), float(valid_geo['Latitude'].max())
    lon_min, lon_max = float(valid_geo['Longitude'].min()), float(valid_geo['Longitude'].max())
    lat_padding = max((lat_max - lat_min) * .1, .02)
    lon_padding = max((lon_max - lon_min) * .1, .02)
    figure.update_geos(
        center={'lat': (lat_min + lat_max) / 2, 'lon': (lon_min + lon_max) / 2},
        lataxis_range=[max(-90, lat_min - lat_padding), min(90, lat_max + lat_padding)],
        lonaxis_range=[max(-180, lon_min - lon_padding), min(180, lon_max + lon_padding)],
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor='#F0F0EA',
        showocean=True,
        oceancolor='#E8EFEB',
        countrycolor='#C9CEC6',
        coastlinecolor='#C9CEC6',
        bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(style_figure(figure, 520), width='stretch')
    if sampled:
        st.caption(f'Displayed a deterministic sample of {MAX_MAP_POINTS:,} from {len(valid_geo):,} filtered valid coordinates.')
    if severity_column:
        st.caption('Marker colors identify source severity codes; the supplied dataset does not contain descriptive code labels.')

    section_header(
        'Geographic summary',
        'Coverage bounds, category counts, and coordinate-cell counts use all filtered valid records, not only the plotted sample.',
    )
    summary_cols = st.columns([1, 1.35, 1])
    if 'Disaster_Type' in valid_geo:
        counts = valid_geo['Disaster_Type'].value_counts().sort_index()
        summary_cols[0].dataframe(
            pd.DataFrame({'Disaster category': [f'Code {value}' for value in counts.index], 'Observations': counts.values}),
            width='stretch',
            hide_index=True,
        )
    coordinate_cells = (
        valid_geo.assign(
            Latitude_cell=valid_geo['Latitude'].round(2),
            Longitude_cell=valid_geo['Longitude'].round(2),
        )
        .groupby(['Latitude_cell', 'Longitude_cell'])
        .size()
        .rename('Observations')
        .reset_index()
        .sort_values('Observations', ascending=False)
        .head(10)
    )
    summary_cols[1].dataframe(
        coordinate_cells.rename(columns={
            'Latitude_cell': 'Latitude (rounded)',
            'Longitude_cell': 'Longitude (rounded)',
        }),
        width='stretch',
        hide_index=True,
    )
    summary_cols[2].write(f'Invalid or missing coordinates excluded: **{invalid_count:,} / {len(df):,}**')
    summary_cols[2].caption('Coordinate cells use rounded source coordinates; they are not named locations.')
    st.caption('The map is a visualization of the supplied dataset only; location accuracy and incident authenticity are not independently validated.')


if __name__ == '__main__':
    main()
