from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px


def dataset_summary(df: pd.DataFrame) -> Dict[str, float | int | None]:
    injury_records = int((df['Injury_Detection'] == 1).sum()) if 'Injury_Detection' in df else None
    summary = {
        'records': int(len(df)),
        'drones': int(df['Drone_ID'].nunique()) if 'Drone_ID' in df.columns else None,
        'disaster_types': int(df['Disaster_Type'].nunique()) if 'Disaster_Type' in df.columns else None,
        'injury_detection_code_1_records': injury_records,
        'avg_priority': round(float(pd.to_numeric(df['Rescue_Priority_Score'], errors='coerce').mean()), 2) if 'Rescue_Priority_Score' in df.columns else None,
        'avg_battery': round(float(pd.to_numeric(df['Battery_Level_percent'], errors='coerce').mean()), 2) if 'Battery_Level_percent' in df.columns else None,
        'severity_distribution': df['Injury_Severity'].value_counts().to_dict() if 'Injury_Severity' in df.columns else {},
        'disaster_distribution': df['Disaster_Type'].value_counts().to_dict() if 'Disaster_Type' in df.columns else {},
    }
    return summary


def category_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df:
        return pd.DataFrame(columns=['category', 'records', 'share_percent'])
    counts = df[column].value_counts(dropna=False).sort_index()
    result = pd.DataFrame({
        'category': [f'Code {value}' if pd.notna(value) else 'Missing' for value in counts.index],
        'records': counts.to_numpy(dtype='int64'),
    })
    total = max(int(len(df)), 1)
    result['share_percent'] = result['records'] / total * 100
    return result


def build_disaster_charts(df: pd.DataFrame, include_map: bool = False):
    charts = {}
    if 'Disaster_Type' in df.columns:
        disaster_counts = category_distribution(df, 'Disaster_Type')
        charts['disaster_type'] = px.pie(disaster_counts, names='category', values='records', title='Disaster distribution', hole=0.35)
    if 'Injury_Severity' in df.columns:
        severity_counts = category_distribution(df, 'Injury_Severity')
        charts['severity'] = px.bar(severity_counts, x='category', y='records', title='Injury severity distribution')
    if 'Timestamp' in df.columns:
        timestamps = pd.to_datetime(df['Timestamp'], errors='coerce').dropna()
        monthly = (
            timestamps.dt.to_period('M')
            .value_counts()
            .sort_index()
            .rename_axis('month')
            .reset_index(name='records')
        )
        monthly['month'] = monthly['month'].dt.to_timestamp()
        charts['timeline'] = px.line(monthly, x='month', y='records', markers=True, title='Operational timeline')
    if include_map and {'Latitude', 'Longitude'}.issubset(df.columns):
        map_columns = ['Latitude', 'Longitude']
        for column in ['Disaster_Type', 'Injury_Severity']:
            if column in df.columns:
                map_columns.append(column)
        map_data = df[map_columns].copy()
        map_data['Latitude'] = pd.to_numeric(map_data['Latitude'], errors='coerce')
        map_data['Longitude'] = pd.to_numeric(map_data['Longitude'], errors='coerce')
        map_data = map_data.dropna(subset=['Latitude', 'Longitude'])
        charts['map'] = px.scatter_geo(
            map_data,
            lat='Latitude',
            lon='Longitude',
            color='Injury_Severity' if 'Injury_Severity' in map_data.columns else None,
            hover_name='Disaster_Type' if 'Disaster_Type' in map_data.columns else None,
            projection='natural earth',
            title='Disaster location map',
        )
    return charts


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ['Disaster_Type', 'Injury_Severity', 'Injury_Detection', 'Drone_Operational_Status']:
        if column in df.columns:
            counts = df[column].value_counts().reset_index()
            counts.columns = ['value', 'count']
            counts['value'] = counts['value'].map(
                lambda value: f'Code {value}' if pd.notna(value) else 'Missing'
            )
            rows.append(counts.assign(metric=column))
    if not rows:
        return pd.DataFrame(columns=['metric', 'value', 'count'])
    return pd.concat(rows, ignore_index=True)
