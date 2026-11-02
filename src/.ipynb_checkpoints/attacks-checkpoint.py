import torch
import torch.nn as nn

def fgsm(model, X, y, eps):
    """Nudge every feature by eps in the direction that increases the loss."""
    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(X_adv), y)
    model.zero_grad()
    loss.backward()
    return (X_adv + eps * X_adv.grad.sign()).detach()