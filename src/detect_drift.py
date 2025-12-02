import pandas as pd
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
import numpy as np # Still needed for data simulation

def detect_goal_drift(reference_data_path, current_data_path):
    """
    Detects goal drift by comparing the distribution of agent predictions
    between the reference (baseline) period and the current production data.
    """
    # Load dataframes (simulated production logs)
    ref_df = pd.read_csv(reference_data_path)
    current_df = pd.read_csv(current_data_path)

    # --- Drift Detection Report (Focus ONLY on Target/Output Drift) ---
    drift_report = Report(metrics=[
        TargetDriftPreset(
            # 'Prediction' here is the agent's decision/goal 
            target='agent_prediction_label' 
        )
        # Removed DataDriftPreset since we are ignoring internal features
    ])

    drift_report.run(
        reference_data=ref_df,
        current_data=current_df,
        column_mapping={
            'target': 'agent_prediction_label', # The outcome we are monitoring
        }
    )

    # Check the result of the Target Drift
    target_drift_metric = drift_report.get_dict()['metrics'][0]['result']['target_drift']
    is_drifted = target_drift_metric['drift_detected']
    # If the target is categorical (like ours), the score is usually a distance measure (like Hellinger)
    drift_score = target_drift_metric['drift_score'] 

    print(f"\n--- Goal Drift Detection Summary ---")
    print(f"Goal Drift Detected (on agent_prediction_label): {is_drifted}")
    print(f"Drift Score (Distance between goal distributions): {drift_score:.4f}")

    report_path = "reports/drift_report.html"
    drift_report.save_html(report_path)
    print(f"Saved detailed drift report to {report_path}")

    return is_drifted, drift_score

if __name__ == '__main__':
    # Simulating data that ONLY contains the essential logging fields
    
    # 1. Baseline Reference Data (e.g., 50% Refund, 50% Support)
    ref_data = {
        'agent_prediction_label': ['Refund_Process'] * 50 + ['Technical_Support'] * 50,
        'request_id': [str(np.random.randint(10000)) for _ in range(100)] 
    }
    pd.DataFrame(ref_data).to_csv("data/reference_logs.csv", index=False)
    
    # 2. Current Production Data (e.g., 80% Refund, 20% Support -> Goal Drift!)
    current_data = {
        'agent_prediction_label': ['Refund_Process'] * 80 + ['Technical_Support'] * 20,
        'request_id': [str(np.random.randint(10000)) for _ in range(100)] 
    }
    pd.DataFrame(current_data).to_csv("data/current_logs.csv", index=False)
    
    # Run the detection
    detect_goal_drift("data/reference_logs.csv", "data/current_logs.csv")