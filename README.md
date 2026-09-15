# English → Amharic Neural Machine Translation

A complete end-to-end NMT system that translates English sentences into Amharic
using two LSTM-based sequence-to-sequence models.

| Model | Architecture |
|---|---|
| **Basic Seq2Seq + LSTM** | Encoder LSTM → fixed context vector → Decoder LSTM |
| **Attention-Based Seq2Seq + LSTM** | Encoder LSTM → Bahdanau attention → Decoder LSTM |

---

## Project Structure

```
.
├── 01_dataset_preprocessing.ipynb   # Dataset download, cleaning, vocab building
├── 02_seq2seq_lstm_full.ipynb       # Basic Seq2Seq LSTM training
├── 03_attention_lstm_full.ipynb     # Attention-Based Seq2Seq LSTM training
├── 04_evaluation_comparison.ipynb   # BLEU, chrF, comparison table
├── 05_error_attention_analysis.ipynb# Error analysis & attention heatmaps
├── app.py                           # Streamlit translation application
├── requirements.txt                 # Pinned Python dependencies
├── data/
│   └── processed/
│       ├── train.csv                # 80% split
│       ├── validation.csv           # 10% split
│       └── test.csv                 # 10% split
├── models/
│   ├── en_vocab.json                # English word → index vocabulary
│   ├── amh_vocab.json               # Amharic word → index vocabulary
│   ├── seq2seq_lstm.pt              # Trained Basic Seq2Seq weights
│   ├── seq2seq_lstm_best.pt         # Best checkpoint (lowest val loss)
│   ├── seq2seq_lstm_config.json     # Hyperparameters & metrics
│   ├── attention_lstm.pt            # Trained Attention model weights
│   ├── attention_lstm_best.pt       # Best checkpoint (lowest val loss)
│   └── attention_lstm_config.json   # Hyperparameters & metrics
└── results/
    ├── attention/                   # Attention heatmap PNG files
    ├── model_comparison.csv         # Side-by-side metric comparison
    ├── combined_translation_examples.csv
    ├── evaluation_results.json
    ├── error_analysis_summary.csv
    └── *.png                        # Training curves, charts
```

---

## Dataset

| Property | Value |
|---|---|
| **Name** | `michsethowusu/english-amharic_sentence-pairs_mt560` |
| **Source** | Hugging Face Datasets Hub |
| **License** | CC BY 4.0 |
| **Raw size** | 669,145 aligned sentence pairs |
| **Domain** | Religious, news, and general texts |
| **Languages** | English (source) → Amharic (target) |

After preprocessing (deduplication, length filtering 1–80 tokens):

| Split | Pairs |
|---|---:|
| Train | ~535,000 |
| Validation | ~67,000 |
| Test | ~67,000 |

---

## Installation

### 1. Clone / download the project

```bash
git clone <repo-url>
cd <project-folder>
```

### 2. Create a conda environment (recommended)

```bash
conda create -n amharic_nmt python=3.11 -y
conda activate amharic_nmt
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** replace the `torch` line in `requirements.txt` with the
> CUDA-enabled wheel for your driver, e.g.:
> ```bash
> pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Execution Order

Run the notebooks **in order** from a Jupyter session:

```bash
jupyter lab
```

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01_dataset_preprocessing.ipynb` | Downloads dataset, cleans data, builds vocabularies, saves splits |
| 2 | `02_seq2seq_lstm_full.ipynb` | Trains Basic Seq2Seq LSTM, saves model weights and metrics |
| 3 | `03_attention_lstm_full.ipynb` | Trains Attention LSTM, saves model, attention weights, heatmaps |
| 4 | `04_evaluation_comparison.ipynb` | Computes BLEU / chrF / test loss, generates comparison table |
| 5 | `05_error_attention_analysis.ipynb` | Error analysis, attention visualisations, qualitative examples |

---

## Running the Translation App

After training (steps 1–3 above), launch the Streamlit app:

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

### App features

- **Model selector** — switch between Basic Seq2Seq and Attention-LSTM
- **Example sentences** — pick from pre-loaded examples or type your own
- **JSON output toggle** — shows API-compatible output:

```json
{
  "text": "I am going to the university.",
  "translation": "ወደ ዩኒቨርሲቲ እሄዳለሁ።",
  "model": "Attention-Based Seq2Seq + LSTM",
  "inference_ms": 12.34
}
```

- **Attention heatmap** — visual alignment between source and target tokens
- **Side-by-side comparison** — translate with both models simultaneously

---

## Model Architecture

### Basic Seq2Seq + LSTM

```
English tokens
    │
    ▼
Embedding (256)
    │
    ▼
Encoder LSTM (hidden=512, layers=1)
    │
    ▼  final (hidden, cell) state
Decoder LSTM (hidden=512, layers=1)  ◄── target token at each step
    │
    ▼
Linear → softmax → Amharic token
```

### Attention-Based Seq2Seq + LSTM (Bahdanau)

```
English tokens
    │
    ▼
Embedding (256)
    │
    ▼
Encoder LSTM → all hidden states h₁…hₙ
    │
    ▼  at each decoder step t:
    e_{t,i} = v^T tanh(W_h·hᵢ + W_s·s_{t-1})
    α_{t,i} = softmax(e_{t,i})
    cₜ      = Σ αᵢ·hᵢ          (context vector)
    │
    ▼
Decoder LSTM input = [embedding ; cₜ]
    │
    ▼
Linear (dec_hid + enc_hid + emb) → Amharic token
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Embedding size | 256 |
| Hidden units | 512 |
| Attention dim | 256 |
| Encoder layers | 1 |
| Decoder layers | 1 |
| Dropout | 0.2 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss (ignore `<pad>`) |
| Teacher forcing | 0.5 |
| Gradient clipping | 1.0 |
| Max sequence length | 40 tokens |
| Epochs | 15 |
| Weight initialisation | Xavier uniform |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **BLEU** | n-gram precision with brevity penalty (sacrebleu corpus BLEU) |
| **chrF** | Character n-gram F-score — more suitable for morphologically rich Amharic |
| **Test loss** | Cross-entropy on the held-out test set |
| **Training time** | Wall-clock time for all epochs (minutes) |
| **Inference time** | Average milliseconds per sentence (100-sentence sample) |
| **Parameters** | Total trainable parameters |

Results are saved to `results/model_comparison.csv` and
`results/evaluation_results.json` after running notebook 04.

---

## Key Results (example — actual values depend on training run)

| Metric | Basic Seq2Seq | Attention-LSTM |
|---|---:|---:|
| BLEU | — | — |
| chrF | — | — |
| Test Loss | — | — |
| Training Time (min) | — | — |
| Avg Inference (ms) | — | — |
| Parameters | — | — |

> Fill in after running notebooks 02–04.

---

## Output Files

After running all notebooks the following files are produced:

```
models/
  seq2seq_lstm.pt                 ← final Seq2Seq model weights
  attention_lstm.pt               ← final Attention model weights
  en_vocab.json                   ← English vocabulary (word → index)
  amh_vocab.json                  ← Amharic vocabulary (word → index)
  seq2seq_lstm_config.json        ← hyperparameters & metrics
  attention_lstm_config.json      ← hyperparameters & metrics

results/
  seq2seq_training_curves.png
  attention_training_curves.png
  seq2seq_translation_examples.csv
  attention_lstm_translation_examples.csv
  combined_translation_examples.csv   ← Source|Reference|S2S|Attn
  model_comparison.csv
  model_comparison_chart.png
  evaluation_results.json
  bleu_distribution.png
  error_analysis_summary.csv
  error_category_comparison.png
  bleu_vs_length.png
  attention/
    attention_example_1.png … attention_example_8.png
    attention_long_sentence.png
    attention_weights_examples.json
```

---

## Tokenisation

Both models use **word-level tokenisation**:

```python
# English (lowercased)
re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

# Amharic (case-preserved)
re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
```

Special tokens: `<pad>=0`, `<unk>=1`, `<sos>=2`, `<eos>=3`  
Vocabulary built from **training set only** with `min_freq=2`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: models/seq2seq_lstm.pt` | Run notebooks 01–03 first |
| `ModuleNotFoundError: sacrebleu` | `pip install sacrebleu==2.4.2` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in the training notebooks (try 32) |
| Streamlit app shows blank page | Ensure you are in the project root: `streamlit run app.py` |
| Very low BLEU scores | Increase `N_EPOCHS` (try 20–30) or use the full dataset (`TRAIN_SIZE = None`) |
| Amharic text not rendering | Install a font that supports Ethiopic script (e.g. Noto Serif Ethiopic) |

---

## License

- **Code:** MIT License  
- **Dataset:** CC BY 4.0 — [michsethowusu/english-amharic_sentence-pairs_mt560](https://huggingface.co/datasets/michsethowusu/english-amharic_sentence-pairs_mt560)
