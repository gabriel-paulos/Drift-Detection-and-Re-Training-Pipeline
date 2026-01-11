import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
import mlflow
import os

# Import PEFT components
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from src.llm_decision_layer import load_llm_head, evaluate_model, LLMDecisionHead
# Assuming prepare_data_for_retraining is still defined at the bottom or in a helper file

import dspy

# --- LoRA Configuration ---
LORA_R = 8              # The rank of the update matrices
LORA_ALPHA = 16         # Scaling factor
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["classifier.0"] # Targeting the final classification linear layer

def retrain_agent_with_lora(new_data_path="data/new_alignment_data.csv", output_path="model/retrained_llm_head.pt"):
    """
    Loads the current LLM decision head and fine-tunes it on new, labeled 
    alignment data using LoRA to correct goal drift.
    """
    mlflow.set_experiment("LLM_Agent_Goal_Drift_Pipeline")
    
    # 1. Load Model and Data
    # We load our 'base' model which is the decision head of the LLM agent
    base_model, tokenizer = load_llm_head(path="model/final_llm_head.pt")
    
    # Freeze the base model and prepare it for LoRA/PEFT (standard practice)
    # In a real scenario, you'd load the full base LLM here.
    model = base_model 

    # 2. Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS  # Sequence Classification task
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    print("--- LoRA Model Configuration ---")
    model.print_trainable_parameters() # Shows only the small percentage of parameters being trained
    
    # 3. Setup Training Data
    # Prepare the new, human-aligned dataset
    train_dataloader = DataLoader(
        prepare_data_for_retraining(new_data_path, tokenizer), 
        batch_size=8, 
        shuffle=True
    )
    
    # 4. Fine-Tuning Loop (LoRA Training)
    with mlflow.start_run(run_name="Goal_Drift_LoRA_Retraining") as run:
        
        # Optimizer and Scheduler (only training the LoRA weights)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        num_epochs = 2
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Simplified Training Loop:
        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                # Actual PyTorch/Transformer training steps (loss calculation, backward, step)
                # This step updates ONLY the tiny set of LoRA parameters
                # ...
                # loss.backward()
                # optimizer.step()
                # scheduler.step()
                pass
        
        # 5. Evaluation and Logging
        new_accuracy = evaluate_model(model, train_dataloader)
        mlflow.log_metric("retraining_accuracy_lora", new_accuracy)
        
        # 6. Save and Register the LoRA adapter (much smaller than the full model!)
        # Save the full model with the LoRA weights merged (optional, but cleaner for deployment)
        final_model = model.merge_and_unload()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(final_model.state_dict(), output_path)
        
        # Register the new model version
        mlflow.pytorch.log_model(
            pytorch_model=final_model,
            artifact_path="goal_aligned_lora_head",
            registered_model_name="Goal_Aligned_LLM_Head"
        )
        print(f"LoRA retraining complete. New model saved and registered.")

    return new_accuracy

#Alternatively we could use something like DSPy to prompt optimize to re-align model, will see what difference it makes
class AgentGoalModule(dspy.Signature):
    """Classify user intent into Refund, Support, or Security."""
    input_text = dspy.InputField()
    goal_label = dspy.OutputField(desc="Refund_Process, Technical_Support, or New_Security_Goal")

class CoTAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(AgentGoalModule)

    def forward(self, input_text):
        return self.predictor(input_text=input_text)

# 2. Retraining Loop (Replacing your PyTorch loop)
def dspy_retrain(drifted_examples):
    # drifted_examples = data from your 'current_logs.csv'
    optimizer = dspy.BootstrapFewShot(metric=your_accuracy_metric)
    
    # This "compiles" a new version of the agent that has 'learned' 
    # the new goals from the drifted data
    optimized_agent = optimizer.compile(CoTAgent(), trainset=drifted_examples)
    
    # Save the 'optimized prompt' instead of a .pt model
    optimized_agent.save("model/optimized_agent.json")


if __name__ == '__main__':
    # Placeholder/Dummy Data Prep (as defined previously)
    import numpy as np
    from transformers import AutoTokenizer
    LLM_BACKBONE = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(LLM_BACKBONE)
    
    def prepare_data_for_retraining(path, tokenizer):
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, path, tokenizer):
                self.data = pd.read_csv(path)
                self.tokenizer = tokenizer
            def __len__(self): return len(self.data)
            def __getitem__(self, idx):
                text = self.data.iloc[idx]['text']
                label = self.data.iloc[idx]['label']
                tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
                return {
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze(),
                    'labels': torch.tensor(label)
                }
        return DummyDataset(path, tokenizer)

    # Simulate the "Human-in-the-Loop" labeled data
    data = {
        'text': ["Refund for item X is a priority goal now.", "Urgent support request, needs goal re-alignment.", "New goal: prioritize security.", "Legacy request for refund."],
        'label': [0, 1, 2, 0] # 0=Refund, 1=Support, 2=New Security Goal
    }
    pd.DataFrame(data).to_csv("data/new_alignment_data.csv", index=False)
    
    # Run the LoRA retraining
    # NOTE: You'll need a dummy initial model file at 'model/final_llm_head.pt' 
    # if you run this script standalone.
    # retrain_agent_with_lora()
