# Configuration pour Evidently AI
data_drift_config = {
    "target_drift": {"column_mapping": {"target": "Churn"}},
    "data_drift": {
        "columns": ["tenure", "MonthlyCharges", "TotalCharges"],
        "drift_detection_method": "chisquare"
    }
}
# En production, on utiliserait ceci pour générer des rapports HTML
# report = Report(metrics=[DataDriftPreset()])
# report.run(reference_data=ref, current_data=curr)