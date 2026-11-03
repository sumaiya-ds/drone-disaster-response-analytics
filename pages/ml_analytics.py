from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import (
    COLORS,
    clear_model_evaluation_cache,
    evaluate_models,
    metric_cards,
    page_header,
    section_header,
    style_figure,
)


def _comparison_table(results: dict) -> pd.DataFrame:
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
        'ML Analytics',
        'Compare a baseline against a conservative pre-assessment feature set using a stratified holdout evaluation.',
        eyebrow='MODEL EVALUATION',
    )
    if df is None or df.empty:
        st.info('No dataset is loaded. Select a valid source in Data Workbench.')
        return
    available_targets = [column for column in ('Injury_Severity', 'Injury_Detection') if column in df]
    if not available_targets:
        st.warning('No supported classification target is present in the loaded dataset.')
        return

    controls = st.columns([1.2, 1, 1.4])
    target = controls[0].selectbox('Target variable', available_targets)
    if controls[1].button('Retrain models', type='primary'):
        clear_model_evaluation_cache()
        st.session_state.pop('model_evaluation', None)
    try:
        results = evaluate_models(df, target)
    except (ValueError, KeyError) as exc:
        st.error(f'Model evaluation could not run for {target}: {exc}')
        return
    st.session_state['model_evaluation'] = results

    section_header('Evaluation comparison', 'The target column is excluded from every feature set. All preprocessing is fitted within each training fold.')
    comparison = _comparison_table(results)
    st.dataframe(
        comparison.style.format({
            'Accuracy': '{:.3f}',
            'Balanced accuracy': '{:.3f}',
            'Macro F1': '{:.3f}',
        }),
        width='stretch',
        hide_index=True,
    )
    st.caption(
        f"Stratified holdout · {results['split']['train_records']:,} train / "
        f"{results['split']['test_records']:,} test records · random state {results['random_state']}."
    )

    section_header('Feature-set definitions', 'The baseline removes known target and downstream fields; the leakage-aware set limits predictors to pre-assessment context and telemetry.')
    baseline_excluded = results['feature_sets']['baseline']['excluded_features']
    aware_excluded = results['feature_sets']['leakage_aware']['excluded_features']
    st.markdown(results['leakage_audit']['exclusion_rationale'])
    if results['leakage_audit']['target_in_baseline_features'] or results['leakage_audit']['target_in_leakage_aware_features']:
        st.error('The selected target unexpectedly appears in a feature set. Do not interpret these evaluations.')
    else:
        st.success(f'{target} is excluded from all predictor columns in both feature sets.')
    exact_copies = results['leakage_audit']['exact_target_copy_columns']
    if exact_copies:
        st.warning(f'Columns identical to the selected target were detected: {", ".join(exact_copies)}.')
    else:
        st.caption('No other dataset column is an exact row-by-row copy of the selected target.')
    audit_col1, audit_col2 = st.columns(2)
    with audit_col1:
        st.markdown('**Baseline exclusions**')
        st.caption(', '.join(baseline_excluded) or 'None')
    with audit_col2:
        st.markdown('**Additional conservative exclusions**')
        st.caption(', '.join(aware_excluded) or 'None')
    relationships = pd.DataFrame(results['leakage_audit']['feature_relationships'])
    if not relationships.empty:
        relationships = relationships.rename(columns={
            'feature': 'Candidate feature',
            'association_statistic': 'Association measure',
            'association_value': 'Observed association',
            'exact_target_copy': 'Exact copy of target',
            'timing_documented': 'Collection timing documented',
        })
        st.dataframe(
            relationships.style.format({'Observed association': '{:.4f}'}),
            width='stretch',
            hide_index=True,
        )
        st.caption(
            "Cramer's V is used for encoded categories and Eta for numeric measurements. "
            'These univariate associations are descriptive, not causal and are not used to select features.'
        )
    st.info(
        'The timestamps for injury-specific measurements are not documented. The leakage-aware comparison '
        'therefore excludes injury type, blood-loss severity, pain, and vital signs as a conservative '
        'point-in-time choice—not as a claim that the dataset proves these fields encode the target.'
    )
    with st.expander('Why did the earlier evaluation report 100%?'):
        st.write(
            'The earlier feature-preparation function removed the selected target from its exclusion set. '
            'For Injury_Severity this made the target column itself a model input. Both models then copied '
            'the answer during evaluation. The corrected pipeline always excludes the target and fits imputation, '
            'encoding, and scaling only on training data.'
        )

    selected_set = controls[2].selectbox(
        'Inspect feature set',
        options=['leakage_aware', 'baseline'],
        format_func=lambda value: 'Leakage-aware' if value == 'leakage_aware' else 'Baseline',
    )
    selected_details = results['feature_sets'][selected_set]
    model_names = list(selected_details['results'])
    selected_model = st.selectbox(
        'Inspect classifier',
        model_names,
        index=model_names.index('random_forest') if 'random_forest' in model_names else 0,
        format_func=lambda value: value.replace('_', ' ').title(),
    )
    model_metrics = selected_details['results'][selected_model]
    report = model_metrics['classification_report']
    metric_cards([
        {'label': 'ACCURACY', 'value': f"{model_metrics['accuracy']:.3f}", 'note': 'overall correct predictions', 'accent': COLORS['green']},
        {'label': 'BALANCED ACCURACY', 'value': f"{model_metrics['balanced_accuracy']:.3f}", 'note': 'mean recall across classes', 'accent': COLORS['blue']},
        {'label': 'MACRO F1', 'value': f"{model_metrics['macro_f1']:.3f}", 'note': 'unweighted class F1 mean', 'accent': COLORS['amber']},
    ])

    section_header('Confusion matrix and per-class report', 'Labels use source class codes because the dataset does not provide descriptive mappings.')
    matrix_col, report_col = st.columns([1, 1.15])
    labels = [str(value) for value in model_metrics['labels']]
    confusion = pd.DataFrame(model_metrics['confusion_matrix'], index=labels, columns=labels)
    figure = px.imshow(
        confusion,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=['#F1F1EB', '#A8BDAA', COLORS['green']],
        labels={'x': 'Predicted class code', 'y': 'Actual class code', 'color': 'Records'},
        title=f'{selected_model.replace("_", " ").title()} confusion matrix',
    )
    matrix_col.plotly_chart(style_figure(figure, 350), width='stretch')
    class_report = pd.DataFrame({
        label: {
            'Precision': report[label]['precision'],
            'Recall': report[label]['recall'],
            'F1': report[label]['f1-score'],
            'Support': report[label]['support'],
        }
        for label in labels
    }).T
    report_col.dataframe(class_report.style.format({
        'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}', 'Support': '{:.0f}',
    }), width='stretch')

    section_header('Feature importance', 'Random Forest impurity-based importance for the selected feature set.')
    importance = model_metrics.get('feature_importance')
    if importance:
        importance_frame = pd.DataFrame([
            {'Feature': name.replace('numeric__', '').replace('categorical__', ''), 'Importance': value}
            for name, value in importance.items()
        ]).sort_values('Importance', ascending=True)
        figure = px.bar(
            importance_frame,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Most influential transformed features',
            color_discrete_sequence=[COLORS['green']],
        )
        st.plotly_chart(style_figure(figure, 380), width='stretch')
    else:
        st.info('Feature importance is unavailable for this estimator.')

    section_header('Single-record prediction', 'Apply the fitted pipeline to a selected record; this is a demonstration, not an intervention recommendation.')
    row_index = st.number_input('Dataset row number', min_value=0, max_value=len(df) - 1, value=0, step=1)
    selected_features = selected_details['feature_names']
    row = df.iloc[[int(row_index)]][selected_features]
    pipeline = selected_details['fitted_models'][selected_model]
    try:
        prediction = pipeline.predict(row)[0]
        observed = df.iloc[int(row_index)][target]
        pred_col, actual_col, match_col = st.columns(3)
        pred_col.metric('Predicted class code', str(prediction))
        actual_col.metric('Observed class code', str(observed))
        match_col.metric('Exact match', 'Yes' if prediction == observed else 'No')
    except (ValueError, KeyError) as exc:
        st.error(f'Prediction failed for the selected record: {exc}')

    st.caption('These metrics describe a single stratified random holdout on a historical/simulated dataset. They are not deployment or clinical evidence.')


if __name__ == '__main__':
    main()
