import torch
from .RPSClassifier import RPSClassifier

def load_model() -> RPSClassifier:
    model = RPSClassifier()
    model.load_state_dict(torch.load('best_model.pth'))

    return model