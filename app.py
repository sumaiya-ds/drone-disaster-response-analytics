from __future__ import annotations

import pandas as pd
import streamlit as st

from config import DEFAULT_DATASET
from components.ui import apply_theme
from src.pipeline import load_and_validate_data

st.set_page_config(page_title='Drone Disaster Analysis', page_icon=':material/radar:', layout='wide')


@st.cache_data(show_spinner='Loading and validating the disaster dataset...')
def load_default_dataset(path: str):
    return load_and_validate_data(path)


if 'processed_data' not in st.session_state:
    try:
        result = load_default_dataset(str(DEFAULT_DATASET))
        st.session_state['processed_data'] = result['clean_data']
        st.session_state['quality_report'] = result['quality_report']
        st.session_state['pipeline_summary'] = result['transformation_summary']
        st.session_state['dataset_profile'] = result['profile']
        st.session_state['active_dataset_source'] = result['source_path']
        st.session_state['processed_dataset_path'] = result['processed_path']
        st.session_state['feature_dataset_path'] = result['feature_path']
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.session_state['processed_data'] = pd.DataFrame()
        st.session_state['dataset_error'] = str(exc)

apply_theme()

if st.session_state.get('dataset_error'):
    st.error(f"Could not load the project dataset: {st.session_state['dataset_error']}")

pages = [
    st.Page('pages/command_center.py', title='Command Center', icon=':material/dashboard:'),
    st.Page('pages/data_workbench.py', title='Data Workbench', icon=':material/database:'),
    st.Page('pages/disaster_analytics.py', title='Disaster Analytics', icon=':material/monitoring:'),
    st.Page('pages/geospatial_intelligence.py', title='Geospatial Intelligence', icon=':material/map:'),
    st.Page('pages/ml_analytics.py', title='ML Analytics', icon=':material/model_training:'),
    st.Page('pages/reports.py', title='Reports', icon=':material/description:'),
]

st.navigation(pages).run()
