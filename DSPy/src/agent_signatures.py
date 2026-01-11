import dspy

class GoalClassifier(dspy.Signature):
    """
    Classify user requests into specific company goals.
    Labels: Refund_Process, Technical_Support, New_Security_Goal
    """
    request_text = dspy.InputField(desc="The user's query or problem statement")
    justification = dspy.OutputField(desc="Brief reasoning for the chosen goal")
    goal_label = dspy.OutputField(desc="Must be one of: Refund_Process, Technical_Support, New_Security_Goal")

class SelfHealingAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought adds step-by-step reasoning automatically
        self.predictor = dspy.ChainOfThought(GoalClassifier)

    def forward(self, request_text):
        return self.predictor(request_text=request_text)