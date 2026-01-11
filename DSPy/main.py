from src.detect_drift import check_for_drift_and_retrain
from src.config import setup_dspy

if __name__ == "__main__":
    setup_dspy()
    
    # In a real app, this would run on a schedule or after X requests
    check_for_drift_and_retrain()