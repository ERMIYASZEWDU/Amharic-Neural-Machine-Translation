"""
app.py — English → Amharic NMT Streamlit Application
=====================================================
Serves both trained models through a browser UI and exposes a
JSON translation endpoint compatible with the project spec:

    POST /translate
    Input:  {"text": "I am going to the university."}
    Output: {"translation": "ወደ ዩኒቨርሲቲ እሄዳለሁ።", "model": "...", ...}

Usage:
    streamlit run app.py
"""

import re
import io
import json
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
MAX_LEN   = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Tokenisers
# ──────────────────────────────────────────────────────────────────────────────
def tokenize_english(text: str):
    return re.findall(r"\w+|[^\w\s]", str(text).strip().lower(), re.UNICODE)


def tokenize_amharic(text: str):
    return re.findall(r"\w+|[^\w\s]", str(text).strip(), re.UNICODE)


def decode_ids(ids, itos: dict) -> str:
    tokens = []
    for i in ids:
        tok = itos.get(int(i), UNK_TOKEN)
        if tok == EOS_TOKEN:
            break
        if tok in (SOS_TOKEN, PAD_TOKEN):
            continue
        tokens.append(tok)
    text = " ".join(tokens)
    text = re.sub(r"\s+([።፣፤,.!?;:])", r"\1", text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn       = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, src):
        _, (h, c) = self.rnn(self.dropout(self.embedding(src)))
        return h, c


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.output_dim = output_dim
        self.embedding  = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn        = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out     = nn.Linear(hid_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, token, h, c):
        emb = self.dropout(self.embedding(token.unsqueeze(0)))
        out, (h, c) = self.rnn(emb, (h, c))
        return self.fc_out(out.squeeze(0)), h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


# ── Attention components ───────────────────────────────────────────────────────
class AttnEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid, dec_hid, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn       = nn.LSTM(emb_dim, enc_hid, num_layers=n_layers,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.fc_h      = nn.Linear(enc_hid, dec_hid)
        self.fc_c      = nn.Linear(enc_hid, dec_hid)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, src):
        out, (h, c) = self.rnn(self.dropout(self.embedding(src)))
        return out, torch.tanh(self.fc_h(h)), torch.tanh(self.fc_c(c))


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid, dec_hid, attn_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, attn_dim)
        self.v    = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_h, enc_out, mask):
        src_len = enc_out.shape[0]
        enc     = enc_out.permute(1, 0, 2)
        dec     = dec_h.unsqueeze(1).repeat(1, src_len, 1)
        energy  = torch.tanh(self.attn(torch.cat((dec, enc), dim=2)))
        attn    = self.v(energy).squeeze(2)
        attn    = attn.masked_fill(mask == 0, -1e10)
        return torch.softmax(attn, dim=1)


class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid, dec_hid, attn_dim,
                 n_layers, dropout, pad_idx):
        super().__init__()
        self.output_dim = output_dim
        self.embedding  = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attention  = BahdanauAttention(enc_hid, dec_hid, attn_dim)
        self.rnn        = nn.LSTM(emb_dim + enc_hid, dec_hid, num_layers=n_layers,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out     = nn.Linear(dec_hid + enc_hid + emb_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, tok, h, c, enc_out, mask):
        emb = self.dropout(self.embedding(tok.unsqueeze(0)))
        aw  = self.attention(h[-1], enc_out, mask)
        ctx = torch.bmm(aw.unsqueeze(1), enc_out.permute(1, 0, 2)).permute(1, 0, 2)
        out, (h, c) = self.rnn(torch.cat((emb, ctx), dim=2), (h, c))
        pred = self.fc_out(torch.cat((out, ctx, emb), dim=2).squeeze(0))
        return pred, h, c, aw


class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, en_pad_idx):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self._en_pad_idx = en_pad_idx

    def create_mask(self, src):
        return (src != self._en_pad_idx).permute(1, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Load resources  (cached so Streamlit only loads once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and vocabularies…")
def load_resources():
    # Vocabularies
    with open(MODEL_DIR / "en_vocab.json",  "r", encoding="utf-8") as f:
        en_vocab = json.load(f)
    with open(MODEL_DIR / "amh_vocab.json", "r", encoding="utf-8") as f:
        amh_vocab = json.load(f)

    en_itos  = {int(v): k for k, v in en_vocab.items()}
    amh_itos = {int(v): k for k, v in amh_vocab.items()}

    en_pad  = en_vocab[PAD_TOKEN];  en_unk  = en_vocab[UNK_TOKEN]
    en_sos  = en_vocab[SOS_TOKEN];  en_eos  = en_vocab[EOS_TOKEN]
    amh_pad = amh_vocab[PAD_TOKEN]; amh_unk = amh_vocab[UNK_TOKEN]
    amh_sos = amh_vocab[SOS_TOKEN]; amh_eos = amh_vocab[EOS_TOKEN]

    INPUT_DIM  = len(en_vocab)
    OUTPUT_DIM = len(amh_vocab)

    # ── Basic Seq2Seq ──
    enc_s2s = Encoder(INPUT_DIM,  256, 512, 1, 0.2, en_pad)
    dec_s2s = Decoder(OUTPUT_DIM, 256, 512, 1, 0.2, amh_pad)
    s2s     = Seq2Seq(enc_s2s, dec_s2s).to(DEVICE)
    s2s.load_state_dict(
        torch.load(MODEL_DIR / "seq2seq_lstm.pt", map_location=DEVICE)
    )
    s2s.eval()

    # ── Attention Seq2Seq ──
    enc_a = AttnEncoder(INPUT_DIM,  256, 512, 512, 1, 0.2, en_pad)
    dec_a = AttnDecoder(OUTPUT_DIM, 256, 512, 512, 256, 1, 0.2, amh_pad)
    attn  = AttentionSeq2Seq(enc_a, dec_a, en_pad).to(DEVICE)
    attn.load_state_dict(
        torch.load(MODEL_DIR / "attention_lstm.pt", map_location=DEVICE)
    )
    attn.eval()

    return {
        "en_vocab":  en_vocab,
        "amh_vocab": amh_vocab,
        "en_itos":   en_itos,
        "amh_itos":  amh_itos,
        "en_pad": en_pad, "en_unk": en_unk, "en_sos": en_sos, "en_eos": en_eos,
        "amh_pad": amh_pad, "amh_unk": amh_unk, "amh_sos": amh_sos, "amh_eos": amh_eos,
        "s2s":  s2s,
        "attn": attn,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def translate_seq2seq(sentence: str, res: dict) -> tuple[str, float]:
    ev = res["en_vocab"]
    src_ids = (
        [res["en_sos"]]
        + [ev.get(t, res["en_unk"]) for t in tokenize_english(sentence)[:MAX_LEN]]
        + [res["en_eos"]]
    )
    src_t = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(1)

    t0 = time.perf_counter()
    h, c = res["s2s"].encoder(src_t)
    tok  = torch.tensor([res["amh_sos"]], dtype=torch.long, device=DEVICE)
    gen  = []
    for _ in range(MAX_LEN):
        out, h, c = res["s2s"].decoder(tok, h, c)
        p = out.argmax(1).item()
        if p == res["amh_eos"]:
            break
        gen.append(p)
        tok = torch.tensor([p], dtype=torch.long, device=DEVICE)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return decode_ids(gen, res["amh_itos"]), elapsed_ms


@torch.no_grad()
def translate_attention(sentence: str, res: dict) -> tuple[str, float, np.ndarray, list]:
    ev  = res["en_vocab"]
    src_tokens = tokenize_english(sentence)[:MAX_LEN]
    src_ids = (
        [res["en_sos"]]
        + [ev.get(t, res["en_unk"]) for t in src_tokens]
        + [res["en_eos"]]
    )
    src_t = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(1)

    t0 = time.perf_counter()
    model    = res["attn"]
    enc_out, h, c = model.encoder(src_t)
    mask = model.create_mask(src_t)
    tok  = torch.tensor([res["amh_sos"]], dtype=torch.long, device=DEVICE)
    gen, attn_rows = [], []
    for _ in range(MAX_LEN):
        out, h, c, aw = model.decoder(tok, h, c, enc_out, mask)
        p = out.argmax(1).item()
        if p == res["amh_eos"]:
            break
        gen.append(p)
        attn_rows.append(aw.squeeze(0).cpu().numpy())
        tok = torch.tensor([p], dtype=torch.long, device=DEVICE)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    translation  = decode_ids(gen, res["amh_itos"])
    attn_matrix  = np.array(attn_rows) if attn_rows else np.zeros((1, len(src_ids)))
    return translation, elapsed_ms, attn_matrix, src_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Attention heatmap (returns a PNG bytes buffer)
# ──────────────────────────────────────────────────────────────────────────────
def render_attention_heatmap(
    src_tokens: list,
    tgt_text: str,
    attn_matrix: np.ndarray,
    title: str = "Bahdanau Attention",
) -> io.BytesIO:
    tgt_tokens = tokenize_amharic(tgt_text) if tgt_text.strip() else ["(empty)"]
    n_trg = min(len(tgt_tokens), attn_matrix.shape[0])
    n_src = min(len(src_tokens), attn_matrix.shape[1])

    if n_trg == 0 or n_src == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No attention to display",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        matrix   = attn_matrix[:n_trg, :n_src]
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix   = matrix / np.maximum(row_sums, 1e-12)

        fig, ax = plt.subplots(
            figsize=(max(6, n_src * 0.65), max(4, n_trg * 0.55))
        )
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

        ax.set_xticks(range(n_src))
        ax.set_xticklabels(src_tokens[:n_src], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_trg))
        ax.set_yticklabels(tgt_tokens[:n_trg], fontsize=9)
        ax.set_xlabel("Source (English) tokens", fontsize=10)
        ax.set_ylabel("Generated Amharic tokens", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Attention weight")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="English → Amharic Translator",
        page_icon="🇪🇹",
        layout="wide",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🇬🇧 → 🇪🇹  English → Amharic Neural Machine Translation")
    st.markdown(
        "Translate English sentences into Amharic using two trained models: "
        "a **Basic Seq2Seq LSTM** and an **Attention-Based Seq2Seq LSTM** (Bahdanau)."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.radio(
            "Select model",
            ["Attention-Based Seq2Seq + LSTM (recommended)", "Basic Seq2Seq + LSTM"],
            index=0,
        )
        show_attention = st.checkbox("Show attention heatmap", value=True)
        show_json      = st.checkbox("Show JSON output (API format)", value=False)

        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown(
            """
**Dataset:** MT560 English–Amharic  
**Architecture:** LSTM encoder–decoder  
**Attention:** Bahdanau additive  
**Vocabulary:** Word-level  
**Max length:** 40 tokens  

Run on: `"""
            + str(DEVICE)
            + "`"
        )

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        res = load_resources()
    except FileNotFoundError as e:
        st.error(
            f"Model files not found: {e}\n\n"
            "Please run notebooks 01–03 first to train the models."
        )
        st.stop()

    # ── Input area ────────────────────────────────────────────────────────────
    st.subheader("📝 Enter English text")

    example_sentences = [
        "I am going to the university.",
        "God created the heavens and the earth.",
        "How are you today?",
        "Ethiopia is a beautiful country.",
        "The students are studying hard for their exams.",
        "I love reading books.",
    ]

    col1, col2 = st.columns([3, 1])
    with col2:
        selected = st.selectbox("Or pick an example", ["— type your own —"] + example_sentences)

    with col1:
        default_text = "" if selected == "— type your own —" else selected
        user_input   = st.text_area(
            "English sentence",
            value=default_text,
            height=100,
            placeholder="Type an English sentence here…",
            label_visibility="collapsed",
        )

    translate_btn = st.button("🔁 Translate", type="primary", use_container_width=False)

    # ── Translation ───────────────────────────────────────────────────────────
    if translate_btn and user_input.strip():
        use_attention = "Attention" in model_choice
        model_name    = "Attention-Based Seq2Seq + LSTM" if use_attention else "Basic Seq2Seq + LSTM"

        with st.spinner("Translating…"):
            if use_attention:
                translation, inf_ms, attn_matrix, src_tokens = translate_attention(
                    user_input, res
                )
            else:
                translation, inf_ms = translate_seq2seq(user_input, res)
                attn_matrix, src_tokens = None, None

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🌍 Translation")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**English (source)**")
            st.info(user_input)
        with r2:
            st.markdown("**Amharic (translation)**")
            st.success(translation if translation else "*(no output generated)*")

        st.caption(
            f"Model: **{model_name}** | Inference time: **{inf_ms:.1f} ms**"
        )

        # ── JSON output ───────────────────────────────────────────────────────
        if show_json:
            st.markdown("---")
            st.subheader("📦 JSON Output (API format)")
            json_output = {
                "text":        user_input,
                "translation": translation,
                "model":       model_name,
                "inference_ms": round(inf_ms, 2),
            }
            st.code(json.dumps(json_output, ensure_ascii=False, indent=2), language="json")

        # ── Attention heatmap ─────────────────────────────────────────────────
        if show_attention and use_attention and attn_matrix is not None:
            st.markdown("---")
            st.subheader("🔍 Attention Heatmap")
            st.markdown(
                "Each row = one generated Amharic token. "
                "Each column = one English source token. "
                "Brighter = higher attention weight."
            )
            heatmap_buf = render_attention_heatmap(
                src_tokens  = src_tokens,
                tgt_text    = translation,
                attn_matrix = attn_matrix,
                title       = "Bahdanau Attention — Source vs. Generated",
            )
            st.image(heatmap_buf, use_container_width=True)

    elif translate_btn:
        st.warning("Please enter an English sentence before clicking Translate.")

    # ── Compare both models ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚖️ Compare both models side by side"):
        cmp_input = st.text_input(
            "English sentence for comparison",
            placeholder="Enter a sentence to compare both models…",
        )
        if st.button("Compare", key="cmp_btn") and cmp_input.strip():
            with st.spinner("Running both models…"):
                s2s_out,  s2s_ms  = translate_seq2seq(cmp_input, res)
                attn_out, attn_ms, attn_mat, src_toks = translate_attention(cmp_input, res)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Basic Seq2Seq + LSTM**")
                st.success(s2s_out or "*(no output)*")
                st.caption(f"Inference: {s2s_ms:.1f} ms")
            with c2:
                st.markdown("**Attention-Based Seq2Seq + LSTM**")
                st.success(attn_out or "*(no output)*")
                st.caption(f"Inference: {attn_ms:.1f} ms")

            if attn_mat is not None:
                st.markdown("**Attention heatmap (Attention model)**")
                buf = render_attention_heatmap(src_toks, attn_out, attn_mat)
                st.image(buf, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<small>English → Amharic NMT Project · "
        "Dataset: MT560 (CC BY 4.0) · "
        "Built with PyTorch & Streamlit</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
