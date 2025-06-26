import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("training_log.csv", header=None)

# Assuming 10 input cols + 10 target cols + 10 predicted cols
seq_len = len(df.columns) // 3
input_cols = list(range(0, seq_len))
target_cols = list(range(seq_len, 2 * seq_len))
pred_cols = list(range(2 * seq_len, 3 * seq_len))

# Select the last row (most recent prediction)
last_row = df.iloc[-1]
inputs = last_row[input_cols].values
targets = last_row[target_cols].values
preds = last_row[pred_cols].values

# Plot
plt.figure(figsize=(10, 5))
plt.plot(inputs, label="Input", marker='o', linestyle='dotted')
plt.plot(targets, label="Target (Sorted)", marker='o')
plt.plot(preds, label="Predicted", marker='x')
plt.title("NTM Sorting Task – Final Prediction")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ntm_sort_plot.png")
plt.show()
