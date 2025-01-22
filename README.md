# LinearVC: Leveraging Self-Supervised Representations for Linear Disentanglement in One-Shot Voice Conversion

LinearVC is an advanced voice conversion model that utilizes self-supervised learning (SSL) representations to achieve linear disentanglement of speech attributes, enabling high-quality one-shot voice conversion. This repository provides the implementation, pretrained models, and necessary scripts for training and inference.

---

## Key Features

- **One-Shot Voice Conversion**: Transforms the speaker identity of a source speech into that of an arbitrary target using only a single utterance.
- **Self-Supervised Representations**: Employs features extracted from SSL models like WavLM for robust content and speaker disentanglement.
- **Linear Disentanglement**: Utilizes K-means quantization and linear operations to effectively separate content and speaker attributes.
- **High-Fidelity Output**: Delivers superior performance across multiple objective and subjective evaluation metrics.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA (for GPU acceleration)

### Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/simyoungjun/LinearVC.git
   cd LinearVC
```
2. **Install dependencies**:
   ```bash
    pip install -r requirements.txt
    ```

### Steps
