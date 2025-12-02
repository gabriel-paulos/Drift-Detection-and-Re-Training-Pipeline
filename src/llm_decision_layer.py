import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score
import os

# --- Configuration ---
# Even though we use DistilBERT as the *mechanism*, we treat it as the final 
# classification layer of a much larger, proprietary LLM agent.
LLM_BACKBONE = "distilbert-base-uncased"
NUM_LABELS = 3 
LABEL_MAP = {
    0: 'Refund_Process', 
    1: 'Technical_Support', 
    2: 'New_Security_Goal' # The category that is susceptible to Goal Drift
}

class LLMDecisionHead(nn.Module):
    """
    Simulates the final classification head of a large LLM agent 
    for goal-oriented decision-making (e.g., routing, intent classification).
    This is the layer that is fine-tuned (via LoRA) to correct Goal Drift.
    """
    def __init__(self, num_labels):
        super().__init__()
        # Use a pre-trained encoder (e.g., DistilBERT) to simulate the 
        # LLM's final feature extraction layer before the decision head.
        self.encoder = AutoModel.from_pretrained(LLM_BACKBONE)
        
        # The classification head (the part we will actually train with LoRA)
        self.classifier = nn.Sequential(
            # Naming the first layer "0" to match the LoRA target configuration
            nn.Linear(self.encoder.config.dim, self.encoder.config.dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.dim, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        """Processes input and returns goal logits."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token output (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

def load_llm_head(path="model/final_llm_head.pt"):
    """
    Loads the decision head model and tokenizer from the specified path.
    If the file does not exist, it starts with an untrained model structure.
    """
    # Ensure the model directory exists for initial saving
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    model = LLMDecisionHead(NUM_LABELS)
    
    # Check for a saved state dict
    if os.path.exists(path):
        try:
            # Note: A real LLM system might load a PEFT adapter here.
            model.load_state_dict(torch.load(path))
            print(f"Loaded LLM Decision Head from {path}")
        except Exception as e:
            print(f"Error loading model state from {path}: {e}")
            print("Starting with untrained weights.")
    else:
        print(f"Model file not found at {path}. Starting with untrained weights.")
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_BACKBONE)
    return model, tokenizer

def evaluate_model(model, data_loader):
    """
    Placeholder for rigorous evaluation against a goal-alignment test suite.
    In a production system, this would calculate metrics like F1 or accuracy.
    """
    model.eval()
    # Simplified evaluation logic for demonstration:
    # (Actual evaluation code would go here)
    
    # Return a simulated high accuracy value after successful retraining
    return 0.92 

if __name__ == '__main__':
    # Example usage:
    model, tokenizer = load_llm_head()
    print(f"Model architecture loaded: {model.__class__.__name__}")
    print(f"Tokenizer loaded: {tokenizer.name_or_path}")

    # Small inference example
    sample_text = "I need help setting up my new security token."
    inputs = tokenizer(sample_text, return_tensors='pt')
    
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        predicted_id = torch.argmax(logits, dim=1).item()
        predicted_label = LABEL_MAP.get(predicted_id)

    print(f"\nSample Query: '{sample_text}'")
    print(f"Predicted Goal: {predicted_label} (ID: {predicted_id})")