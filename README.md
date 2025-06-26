# 🧠 Neural Turing Machine — Haskell Implementation

This project is a **Neural Turing Machine (NTM)** built from scratch in **Haskell**, capable of learning algorithmic tasks using memory and attention. The current task it learns is **sequence sorting** — and it does so beautifully over training.

![Demo Animation](https://github.com/AdvythVaman05/NeuralTuringMachine/blob/4664a4ceace7d968a9ce19351c8c28de87c607aa/cropped_output%20(1).gif)

---

## 🔍 Project Highlights

- ✨ Written entirely in **pure Haskell** (no external ML libraries)
- 🧠 Implements NTM architecture: controller (LSTM), read/write heads, memory matrix
- 🧪 Trains to **sort real-valued vectors** using gradient descent
- 📈 Logs predictions every N epochs
- 📊 Animated **3D Plotly visualization** showing model learning progress

---

## 📦 Project Structure

```
NeuralTuringMachine/
├── app/
│   └── Main.hs                 # Entry point
├── src/
│   ├── Math.hs                 # Tensor and math operations
│   ├── NTM.hs                  # Core NTM logic
│   ├── Train.hs                # Training loop
│   ├── Memory.hs
│   ├── Addressing.hs
│   └── LSTM.hs                 # Controller
└── README.md
```

---

## 🚀 Getting Started

### 🛠️ Build & Run (Haskell)

Make sure you have [Stack](https://docs.haskellstack.org/en/stable/README/) installed:

```bash
stack build
stack run
```

## 💡 Why Haskell?

Haskell is a great choice for neural architecture prototyping due to:

- ✅ Pure functional nature
- ✅ Easy matrix manipulation with custom types
- ✅ Strong type system ensuring safety in complex logic

---

## 🙌 Acknowledgements

- Inspired by DeepMind's original NTM paper (2014)
- Animation powered by Plotly
- Haskell framework bootstrapped with Stack

---

## 🧠 Future Work

- 🧮 Add copy/reverse memory tasks
- 🎯 Enable curriculum learning
- 📚 Export model weights and convert to ONNX

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) © 2025 Advyth Vaman Akalankam.
