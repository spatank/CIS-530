### Part 0

Update the function `average_precision` in `eval.py` to implement the average precision metric.

### Part 1A

Extract training text into files for training language models (`generate_lm_training_text.py`)

### Part 1B

Use the language model training script (`train_lm.py`) to train language models for each condition. You can modify this file to improve the language model.

### Part 1C

Predict subjects' condition using language models. Complete the function `score_subjects` in `predict_lm.py`, and run for the three experiments:

- (`conditionPOS` vs `conditionNEG`)
1. `ptsd` vs `control`
2. `depression` vs `control`
3. `ptsd` vs `depression`

Turn in the output of these experiments.

### Extensions

Update `predict_ext.py` to implement your extensions.