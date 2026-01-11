import dspy
import os
import torch
import pandas as pd
from dspy.teleprompt import BootstrapFinetune
from .agent_signatures import SelfHealingAgent
from .config import MODEL_PATH, TRAIN_PATH, setup_dspy

def run_dspy_retraining(use_peft=True):
    """
    Retrains the agent logic. 
    If use_peft is True, it performs LoRA fine-tuning on a local model.
    """
    setup_dspy()
    
    # 1. Prepare Data
    df = pd.read_csv(TRAIN_PATH)
    trainset = [
        dspy.Example(request_text=row['text'], goal_label=row['label']).with_inputs('request_text')
        for _, row in df.iterrows()
    ]

    # 2. Define Metric
    def metric(gold, pred, trace=None):
        return gold.goal_label.lower() == pred.goal_label.lower()

    if use_peft:
        print("🛠 Starting Deep Retraining (PEFT/LoRA)...")
        # Define the 'Student' model (The one getting the LoRA adapters)
        # In a real setup, this would be a local path or HF ID
        student_model = "meta-llama/Llama-3.2-1B" 
        
        # BootstrapFinetune automatically handles the synthetic data generation
        # and calls the PEFT trainer under the hood.
        tp = BootstrapFinetune(metric=metric)
        
        # This will compile the program and produce a LoRA adapter
        # It requires a GPU in the Docker environment
        compiled_program = tp.compile(
            SelfHealingAgent(), 
            trainset=trainset,
            target=student_model,
            epochs=3,
            bf16=True if torch.cuda.is_bf16_supported() else False
        )
    else:
        print("📝 Starting Prompt Optimization (Few-Shot)...")
        from dspy.teleprompt import BootstrapFewShot
        tp = BootstrapFewShot(metric=metric)
        compiled_program = tp.compile(SelfHealingAgent(), trainset=trainset)

    # 3. Save the resulting 'Program' (JSON configuration + weights reference)
    compiled_program.save(MODEL_PATH)
    print(f"✅ Optimization complete. Saved to {MODEL_PATH}")