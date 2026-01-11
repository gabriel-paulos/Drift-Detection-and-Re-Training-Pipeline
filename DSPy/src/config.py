import dspy
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model/optimized_goal_agent.json")
LOG_PATH = os.path.join(BASE_DIR, "data/current_logs.csv")
REF_PATH = os.path.join(BASE_DIR, "data/reference_logs.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "data/alignment_data.csv")

# LLM Setup
def setup_dspy():
    # You can swap 'openai/gpt-4o-mini' for 'ollama/llama3' or any provider
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)