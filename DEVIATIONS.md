# Deviations

Every choice made here that the paper (*FMNV: A Dataset of Media-Published News
Videos for Fake News Detection*, Wang, Qian & Li, ICIC 2025) does not specify.
Appended to as the reproduction is built.


## Environment
- **Python 3.13 / venv at `.venv/`** — paper states no interpreter or dependency
  versions. Installed: torch 2.13.0, transformers 5.16.1, numpy 2.5.2,
  scikit-learn 1.9.0.
- **Device: `mps` when available, else `cpu`.** The paper does not state hardware;
  CLAUDE.md forbids CUDA since the team is on Macs.

## Co-attention module (`modules/coattention.py`)
The paper gives only equations 5-7 and the authors never uploaded the file, so
the following implementation details are ours:
- **d_k = d_v = d_model / n_heads = 32** (128 / 4). The paper names `d_k` in the
  scaling term but never gives its value; equal splitting is the standard
  Transformer convention.
- **Per-head projections + an output `Linear(n_heads*d_v, d_model)`** after head
  concatenation, as in Vaswani et al. The paper's eq. 5 writes single-head
  attention only.
- **Dropout (0.1, the paper's stated rate) applied to the attention weights and
  to the output projection.** The paper does not say where dropout sits.
- **Post-norm ordering** — `LayerNorm(x + Attn(...))` — taken literally from
  eq. 6-7 rather than the pre-norm variant.
- **Fully-masked context handling:** when a stream has an all-zero attention mask
  (an empty transcript), softmax over all `-inf` would be NaN. Those rows are
  computed unmasked and the attended output is then zeroed, so the sample passes
  through as `LayerNorm(residual)` only. The paper does not discuss empty inputs.
- **`v_len`/`s_len` accept an int, a `(B,)` length tensor, or a `(B, L)` attention
  mask.** The authors' call site does not reveal which was intended.

## Model (`model.py`)
- **`TransformerEncoderLayer` uses PyTorch defaults** for everything the authors'
  file does not set: `dim_feedforward=2048`, `dropout=0.1`, ReLU, post-norm. The
  2048-wide FFN dominates the parameter count (527K of 923K total). CLAUDE.md's
  "~500K params" estimate is therefore low; the architecture as specified gives
  **922,754** parameters.
- **Masked mean pooling** over the token dimension (padded positions excluded)
  rather than a plain mean. A plain mean would dilute every short title with
  padding, and would make an all-zero transcript contribute a spurious constant.
- **`has_transcript` is injected as a single scalar concatenated to the fused
  128-d vector**, giving `Linear(129, 2)`. The paper has no such feature at all;
  this is the leakage probe required by CLAUDE.md. Concatenating at the last
  layer keeps the rest of the network identical between the two runs.
- **`num_classes = 2` with cross-entropy over logits** (not a 1-logit sigmoid),
  matching the authors' `Linear(dim, 2)`.
