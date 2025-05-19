<h1 align="center">🎼 Text2midi-InferAlign</h1>
<p align="center"><b>Improving Symbolic Music Generation with Inference-Time Alignment</b></p>

<div align="center">

  [![Examples](https://img.shields.io/badge/Examples-Demo-blue?style=flat-square&logo=music)](https://amaai-lab.github.io/t2m-inferalign/)
  [![arXiv](https://img.shields.io/badge/arXiv-2406.02255-brightgreen.svg)](https://arxiv.org/abs/2406.02255)
</div>

---

**Text2midi-InferAlign** is an inference-time technique that enhances **symbolic music generation** by improving alignment between generated compositions and textual prompts. It is designed to extend autoregressive models—like **Text2Midi**—without requiring any additional training or fine-tuning.

Our method introduces two lightweight but effective alignment-based objectives into the generation process:

- **🎵 Text-Audio Consistency:** Encourages the temporal structure of the music to reflect the rhythm and pacing implied by the input caption.
- **🎵 Harmonic Consistency:** Penalizes musically inconsistent notes (e.g., out-of-key or dissonant phrases), promoting tonal coherence.

By incorporating these alignment signals into the decoding loop, **Text2midi-InferAlign** produces music that is not only more faithful to textual descriptions but also harmonically robust.

We evaluate our technique on **Text2Midi**, a state-of-the-art text-to-MIDI generation model, and report improvements in both **objective metrics** and **human evaluations**.

---

## 📦 Installation & Usage

This repository contains the implementation of the Inference-Time Alignment module. Follow the steps below to get started.

### 1. Clone the Repository

```bash
git clone https://github.com/AMAAI-Lab/t2m-inferalign.git
cd t2m-inferalign
```

### 2. Set Up the Environment

We recommend using **Python 3.10** and `conda` for environment management.

```bash
conda create -n alignment python=3.10
conda activate alignment
pip install -r requirements.txt
```
Please export your API key. 
```bash
export ANTHROPIC_API_KEY=<your key>
```
or you can set your key [here](https://github.com/AMAAI-Lab/t2m-inferalign/blob/04487795c7a7625ba4d9d17e417774b6a047d19e/progressive_explorer.py#L88C54-L88C71).


### 3. Download Model Weights and Resources

- Download the pretrained **Text2Midi** model from HuggingFace:  
  🔗 https://huggingface.co/amaai-lab/text2midi

- Also download the corresponding tokenizer and soundfonts:  
  🔗 https://huggingface.co/amaai-lab/text2midi/tree/main/

You may choose to organize them like this:

```
t2m-inferalign/
├── checkpoints/
│   └── pytorch_model.bin
├── tokenizer/
│   └── vocab_remi.pkl
├── soundfonts/
│   └── soundfont.sf2
```
Please fix the soundfont path [here](https://github.com/AMAAI-Lab/t2m-inferalign/blob/04487795c7a7625ba4d9d17e417774b6a047d19e/progressive_explorer.py#L31) or [here](https://github.com/AMAAI-Lab/t2m-inferalign/blob/04487795c7a7625ba4d9d17e417774b6a047d19e/progressive_explorer.py#L473).

### 4. Run Inference with Alignment

```bash
python progressive_explorer.py --caption "A gentle piano lullaby with soft melodies" --model_path checkpoints/pytorch_model.bin --tokenizer_path tokenizer/vocab_remi.pkl --output_path outputs/lullaby.mid
```

Optional arguments:
- `--max_tokens`: Max number of tokens in the generated sequence.
- `--batch_size`: Number of tokens to generate before checking rewards.
- `--beams`: Number of parallel sequences to generate.

---

## 📊 Experimental Results

### ✅ Objective Evaluation

We evaluate on the **MidiCaps** dataset using six standard metrics. Our approach outperforms the Text2Midi baseline in all key alignment and tonal consistency metrics.

| Metric                                | Text2Midi | Text2midi-InferAlign |
|---------------------------------------|-----------|-----------------------|
| **CR** (Compression Ratio) ↑          | 2.16      | **2.31**              |
| **CLAP** (Text-Audio Consistency) ↑   | 0.17      | **0.22**              |
| **TB** (Tempo Bin %) ↑                | 29.73     | 35.41                 |
| **TBT** (Tempo Bin w/ Tolerance %) ↑  | 60.06     | 62.59                 |
| **CK** (Correct Key %) ↑              | 13.59     | **29.80**             |
| **CKD** (Correct Key w/ Duplicates %) ↑ | 16.66   | **32.54**             |

> All results are averaged over ~50% of the MidiCaps test set (7913 captions randomly sampled).

---

### 🎧 Subjective Evaluation

A user study was conducted with **24 participants**, comparing outputs from **Text2Midi** and **Text2midi-InferAlign**. Participants rated musical quality and text-audio alignment.

#### Music Quality & Text-Audio Match

| Evaluation Criteria   | Text2Midi (%) | Text2midi-InferAlign (%) |
|-----------------------|---------------|---------------------------|
| Music Quality         | 31.25         | **68.75**                 |
| Text-Audio Match      | 41.67         | **58.33**                 |

#### Caption Type Preference

| Caption Type         | Text2Midi (%) | Text2midi-InferAlign (%) |
|----------------------|---------------|---------------------------|
| MidiCaps Caption     | 48.33         | **51.67**                 |
| Free Text Caption    | 27.78         | **72.22**                 |

These results demonstrate that **Text2midi-InferAlign** significantly enhances both musical structure and semantic relevance, especially for **free-form, open-ended prompts**.

---

## 📌 Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{text2midi-inferalign,
  title={Text2midi-InferAlign: Improving Symbolic Music Generation with Inference-Time Alignment},
  author={Abhinaba Roy, Geeta Puri, Dorien Herremans},
  year={2025},
  url={https://github.com/AMAAI-Lab/t2m-inferalign}
}
```

---

## 🔗 Resources

- 🎧 [Examples](https://amaai-lab.github.io/t2m-inferalign/)
- 🎼 [Text2Midi (Base Model)](https://github.com/AMAAI-Lab/text2midi)
- 🤗 [Text2Midi on HuggingFace](https://huggingface.co/amaai-lab/text2midi)
