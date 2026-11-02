import torch
import torch.nn as nn

def fgsm(model, X, y, eps, mask=None):
    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(X_adv), y)
    model.zero_grad()
    loss.backward()
    step = eps * X_adv.grad.sign()
    if mask is not None:
        step = step * mask
    return (X_adv + step).detach()

def pgd(model, X, y, eps, steps=10, mask=None):
    """Repeated small FGSM steps, always staying within eps of the original."""
    model.eval()
    alpha = 2.5 * eps / steps
    X_adv = X.clone().detach()
    for _ in range(steps):
        X_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(X_adv), y)
        model.zero_grad()
        loss.backward()
        step = alpha * X_adv.grad.sign()
        if mask is not None:
            step = step * mask
        X_adv = X_adv.detach() + step
        X_adv = (X + torch.clamp(X_adv - X, -eps, eps)).detach()
    return X_adv