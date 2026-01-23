# 🏗️ Zero to One: PyTorch Engineering Templates
> **"Don't build the foundation every time you build a house."**

> **"A journey of a thousand miles begins with a single step. This is Step Zero."**

## 📖 Introduction: The Foundation
**Chinese ：忒修斯之船与工程化底座**

This repository provides a **minimalist yet industrial-grade PyTorch project template**.

While the default model implemented here are simple **architecture-based Neural Networks** for classification or regression tasks, the code's value lies **not in the model architecture, but in the engineering structure**.

We believe in the philosophy of the **"Ship of Theseus" (忒修斯之船)**:

1.  **From 0 to 1 (The Base)**: You start with this template. It runs out-of-the-box. It has logs, configs, checkpoints, deterministic seeding, and a clean pipeline without any bloated dependencies.
2.  **From 1 to 100 (The Evolution)**: 
    *   Want to study **Transformers**? Replace `src/model.py`.
    *   Working on **Diffusion**? Modify `src/trainer.py` loop.
    *   Handling 3D Medical Data? Update `src/dataset.py`.
    
Eventually, you might replace every single line of code, turning this ship into a completely new vessel (Mamba, ResNet, LLM, etc.). But this template ensures you start with a **robust hull**, not a pile of loose planks.

---

## ✨ Core Philosophy: Pure PyTorch, Structured

In the gap between "messy single-script tutorials" and "heavy simplified frameworks" (like Lightning/Ignite), this template stands in the middle: **Transparency & Control**.

*   **📂 Modular Design**: Clear separation of concerns. `Model` doesn't know about `Data`; `Trainer` orchestrates everything.
*   **⚙️ Configuration First**: No more "magic numbers" buried in code. All hyperparameters are managed centrally via `configs/config.yaml`.
*   **🛡️ Industrial Robustness**:
    *   **Logging**: Professional logger setup (Console + File).
    *   **Checkpointing**: Auto-save best/last models and support for **Resume Training**.
    *   **Reproducibility**: `seed_everything` for deterministic results.
    *   **Early Stopping**: Prevent overfitting automatically.

---

## 📂 Project Structure Guide

A clear map of the territory (generally, may not specifically cover every model project):

```text
Project_Root/
├── configs/
│   └── config.yaml          # 🧠 The Brain: All hyperparameters (Data, Model, Training)
├── src/
│   ├── dataset.py           # 💿 The Fuel: Data Loading, Preprocessing, Splitting
│   ├── model.py             # ⚙️ The Engine: Model Architecture (The most replaceable part)
│   ├── trainer.py           # 🎮 The Controller: Training Loop, Validation, Checkpointing
│   ├── predict.py           # 🏹 The Output: Inference Logic for production
│   └── utils.py             # 🛠️ The Tools: Config loading, Plots, Seeding, Logging
├── main.py                  # 🔌 The Switch: Entry point that connects all modules
└── README.md
```




## 🌟 Vision: The Engineering Arsenal

Welcome to the **Zero to One PyTorch Engineering Template** project.

This repository acts as a **central library of industrial-grade starting points** for Deep Learning projects. 
We provide standardized, robust, and clean codebases for different architectures. The goal is to let you focus on **Model Innovation (1 to 100)** without worrying about **Project Setup (0 to 1)**.

## 📂 Project Catalog

Each folder in this repository is a self-contained project template focused on a specific architecture.

### 1. [MLP (The Base)](./MLP/)
*   **Architecture**: Multilayer Perceptron (Simple Fully Connected Network).
*   **Purpose**: The **"Hello World"** of our engineering system. It establishes the core standards:
    *   YAML Configuration Management
    *   Singleton Logging
    *   Checkpointing & Resume Logic
    *   Dataset/Dataloader Separation
*   **Status**: ✅ Ready.
*   **Use Case**: Start here if you want to understand the framework or working on tabular data/simple classification.

### 2. Transformer (Planned)
*   **Architecture**: Encoder/Decoder / BERT / GPT-style.
*   **Purpose**: NLP and Sequence Modeling base.
*   **Status**: 🚧 Coming Soon.

### 3. Diffusion (Planned)
*   **Architecture**: UNet / DiT with Gaussian Diffusion.
*   **Purpose**: Generative AI base.
*   **Status**: 🚧 Coming Soon.

---

## 🧭 Why use these templates?

In the current AI landscape, code is often either:
1.  **Too Simple**: Notebooks that fall apart when the project grows.
2.  **Too Complex**: Frameworks (hf, lightning) that hide too much logic behind magic.

The **Zero to One** series strikes the balance:
*   **Pure PyTorch**: No hidden wrappers. You own the training loop.
*   **Industrial Structure**: Pre-built logging, config handling, and reproducibility features.
*   **Copy-Paste Friendly**: Designed to be detached and modified.

## 🚀 How to use

1.  Pick the architecture you need (e.g., `MLP/`).
2.  Copy the folder to your workspace.
3.  Rename it to your project name.
4.  Start coding from `Step 1` (Architecture modification), because `Step 0` (Structure) is already done.



---

## 🗺️ Roadmap & Series

This project is part of a larger initiative to provide **Standardized Engineering Templates** for various Deep Learning architectures. 

Each project folder in this collection will follow this strict structure but focus on a specific architecture, serving as a clean start for that domain:

*   [x] **MLP / Structural Base** (This Project - Classification Task)
*   [ ] **Transformer / LLM Base** (Coming Soon)
*   [ ] **Mamba / State Space Models** (Coming Soon)
*   [ ] **Diffusion / Generative Models** (Coming Soon)
*   [ ] **GNN / Graph Neural Networks** (Coming Soon)

Start simple. Build complex. 

---
*Happy Coding!*


