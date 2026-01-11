import pandas as pd
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from .optimizer import run_dspy_retraining
from .config import REF_PATH, LOG_PATH

def check_for_drift_and_retrain():
    ref_df = pd.read_csv(REF_PATH)
    curr_df = pd.read_csv(LOG_PATH)

    report = Report(metrics=[TargetDriftPreset(target='agent_prediction_label')])
    report.run(reference_data=ref_df, current_data=curr_df)
    
    result = report.get_dict()['metrics'][0]['result']
    drift_detected = result['target_drift']['drift_detected']
    
    if drift_detected:
        print(f"⚠️ DRIFT DETECTED (Score: {result['target_drift']['drift_score']:.4f})")
        run_dspy_retraining()
    else:
        print("✅ No significant drift detected.")