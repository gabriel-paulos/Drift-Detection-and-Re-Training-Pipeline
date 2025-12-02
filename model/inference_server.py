import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import uuid
import time
import random

from src.llm_decision_layer import load_llm_head, NUM_LABELS, LABEL_MAP

# --- Initialize ---
app = FastAPI(title="LLM Goal-Aligned Agent Service")

# Load the model and tokenizer globally (or lazily in a real setup)
# Note: For production, you'd load the model registered via MLflow.
MODEL, TOKENIZER = load_llm_head()

# Simple request schema
class AgentRequest(BaseModel):
    user_query: str

# --- Log Storage (In a real setup, this would write to a Kafka stream or S3) ---
LOG_FILE = "data/production_logs.csv"

# Columns to track for goal drift monitoring
log_data = []

def log_agent_activity(text, label_id, prediction_label):
    """
    Logs the LLM agent's decision for later goal drift monitoring.
    We only log the final decision label, as that reflects the goal outcome.
    """
    log_data.append({
        'timestamp': time.time(),
        'request_id': str(uuid.uuid4()),
        'user_query': text,
        'agent_prediction_id': label_id,
        'agent_prediction_label': prediction_label, # <-- THE CORE DRIFT SIGNAL
        'human_correction_label': None # Placeholder for HiTL feedback
    })
    
    # Simple persistence
    pd.DataFrame(log_data).to_csv(LOG_FILE, index=False)


@app.post("/agent/route_request")
async def route_request(request: AgentRequest):
    """
    The LLM Agent processes the query and outputs a goal-aligned decision.
    """
   # 1. Preprocess (Tokenization)
    encoding = TOKENIZER(
        request.user_query, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    # 2. Inference
    MODEL.eval()
    with torch.no_grad():
        logits = MODEL(encoding['input_ids'], encoding['attention_mask'])
    
    # Get prediction
    predicted_id = torch.argmax(logits, dim=1).item()
    predicted_label = LABEL_MAP.get(predicted_id, "Unknown")
    
    # Log the simplified decision
    log_agent_activity(
        text=request.user_query,
        label_id=predicted_id,
        prediction_label=predicted_label,
    )
    
    return {
        "request_id": log_data[-1]['request_id'],
        "goal_outcome": predicted_label,
        "message": f"Agent routed request to the {predicted_label} process."
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": MODEL is not None}

# Run this server with: uvicorn model.inference_server:app --reload
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)