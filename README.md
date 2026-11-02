# ML-Based Network Intrusion Detection: Attack and Defense

A neural-network intrusion detector trained on NSL-KDD, attacked with adversarial
examples (FGSM and PGD, plus a realistic version limited to features an attacker
could control), then hardened with adversarial training. Includes a Streamlit
dashboard that replays recorded traffic.

## Findings
- The baseline detector is almost fully evaded by very small changes to attack traffic.
- Adversarial training sharply reduces evasion at small attack strengths with almost
  no loss in normal accuracy, but protection fades at larger strengths.
- A realistic attacker, limited to a few controllable features, does far less damage
  at small strengths.
- Performance on the test set is lower than on validation data because the test set
  contains attack types not seen in training.

## Project structure
- `src/` data preparation, model, attacks, training
- `notebooks/` step-by-step experiments (02 to 09)
- `app/dashboard.py` Streamlit demo
- `saved_models/` trained baseline and hardened models

## How to run
1. Clone the repo and create a virtual environment.
2. `pip install -r requirements.txt`
3. Download the dataset into `data/`:

4. Run the dashboard: `streamlit run app/dashboard.py`

## Limitations
Recorded data rather than live traffic. The attacker has full access to the model.
The defense only protects up to the attack strength it was trained on.

Author: Manvendra Singh
