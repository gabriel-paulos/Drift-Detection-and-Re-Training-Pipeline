from langchain.tools import tool
from .agent_signatures import SelfHealingAgent
from .config import MODEL_PATH, setup_dspy
import os

# Initialize and Load the latest optimized program
setup_dspy()
agent_brain = SelfHealingAgent()

if os.path.exists(MODEL_PATH):
    agent_brain.load(MODEL_PATH)

@tool
def company_goal_router(query: str):
    """Useful for identifying which department or goal a user request belongs to."""
    prediction = agent_brain(request_text=query)
    return {
        "identified_goal": prediction.goal_label,
        "reasoning": prediction.justification
    }