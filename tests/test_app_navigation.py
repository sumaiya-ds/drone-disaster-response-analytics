from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ('pages/command_center.py', 'Command Center'),
    ('pages/data_workbench.py', 'Data Workbench'),
    ('pages/disaster_analytics.py', 'Disaster Analytics'),
    ('pages/geospatial_intelligence.py', 'Geospatial Intelligence'),
    ('pages/ml_analytics.py', 'ML Analytics'),
    ('pages/reports.py', 'Reports'),
]


def test_all_navigation_pages_render_with_the_real_dataset():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=180)

    assert not app.exception
    assert len(app.session_state['processed_data']) == 44569

    for page_path, title in PAGES:
        app.switch_page(page_path).run(timeout=180)
        assert not app.exception, f'{page_path} raised: {app.exception}'
        assert any(title in element.value for element in app.title), f'{page_path} did not render its title'
        if page_path == 'pages/ml_analytics.py':
            evaluation = app.session_state['model_evaluation']
            assert not evaluation['leakage_audit']['target_in_baseline_features']
            assert not evaluation['leakage_audit']['target_in_leakage_aware_features']
            assert 'baseline' in evaluation['feature_sets']
            assert 'leakage_aware' in evaluation['feature_sets']
        elif page_path == 'pages/data_workbench.py':
            pipeline = app.session_state['pipeline_summary']
            assert len(pipeline['stage_log']) == 7
            assert pipeline['processing_duration_seconds'] >= 0
        elif page_path == 'pages/disaster_analytics.py':
            assert 'Drone Operational Status' in [item.label for item in app.multiselect]
        elif page_path == 'pages/command_center.py':
            assert any(item.label == 'Observation date range' for item in app.date_input)
