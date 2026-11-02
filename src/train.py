import torch
import torch.nn as nn
from attacks import fgsm

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()

def train_model(model, X, y, X_val, y_val, epochs=10, batch_size=256, lr=1e-3, adv_eps=None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], y[idx]
            if adv_eps is not None:
                xb_adv = fgsm(model, xb, yb, adv_eps)   # fresh attack on the current model
                xb = torch.cat([xb, xb_adv])             # clean + adversarial
                yb = torch.cat([yb, yb])
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch+1}: loss={total/len(X):.4f}  val_acc={accuracy(model, X_val, y_val):.4f}")
    return model