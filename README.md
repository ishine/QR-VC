# LinearVC: Leveraging Self-Supervised Representations for Linear Disentanglement in One-Shot Voice Conversion

One-shot voice conversion (VC) transforms the speaker identity of source speech into that of an arbitrary target using only a single utterance. Recent advancements in self-supervised learning (SSL) models, such as HuBERT and WavLM, have proven effective for one-shot VC tasks due to their ability to independently encode various speech attributes. However, previous disentanglement-based methods leveraging SSL features, such as K-means quantization (KQ) and vector quantization (VQ), primarily focus on extracting discrete content embeddings while suppressing speaker-specific information, underutilizing the full potential of SSL features. Similarly, KNN-VC utilizes SSL features by replacing each source feature with its nearest neighbor from the target speech but requires long target matching sets to ensure phonetic coverage, making it unsuitable for one-shot scenarios. To mitigate this, the Phoneme Hallucinator generates synthetic target data, though at the cost of significant computational overhead and increased model complexity.

In this paper, we propose a novel disentanglement-based one-shot VC model that fully exploits the linear separability of SSL features. By employing K-means quantization and linear operations, our approach effectively disentangles speech attributes, capturing both content and speaker-specific information in a unified framework. Unlike prior methods, it eliminates reliance on long target matching sets or computationally intensive synthetic data generation, achieving high-quality voice conversion through reconstruction losses alone.

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


## Usage

### 1. Data Preprocessing

#### a. Downsample Audio
Use `utils/downsampling.py` to downsample audio files to 16kHz.

```bash
python utils/downsampling.py --in_dir [path_to_original_data] --out_dir [path_to_downsampled_data] --sr [sampling_rate]
```
#### b. Extract SSL Features 
Use `preprocess_ssl.py` to extract features from the 6th layer of WavLM.
```bash
python preprocess_ssl.py --in_dir [path_to_original_data] --out_dir [path_to_downsampled_data] --sr [sampling_rate]

```
### 2. Model Training
Train the LinearVC model using the preprocessed data.
```bash
python train.py --config config/config.json --model_dir [ckpt_save_dir_path] --model [model_name]
```

### 3. Voice Conversion
Perform voice conversion using the trained model.
```bash
python convert.py --config ckptdir/config.json --ptfile [checkpoint_pt_file] --src_path [source.wav] --tgt_path [target.wav] --outdir [convert_output_dir]

```
