# Machine Translation between Chinese and English

This project implements Chinese–English neural machine translation (NMT) and systematically compares three model families:

- RNN-based encoder–decoder with attention
- Transformer trained from scratch
- Pretrained MT5 (google/mt5-small) fine-tuned for Zh→En

All experiments share a unified data pipeline and BLEU-based evaluation.

## Project Structure

- [report.typ](report.typ): Main report in Chinese
- [report_en.typ](report_en.typ): Main report in English
- [data_utils.py](data_utils.py): Data loading, cleaning, tokenization, vocab/SPM helpers
- [rnn_model.py](rnn_model.py): RNN encoder–decoder with attention
- [transformer_model.py](transformer_model.py): Custom Transformer NMT (pos/norm ablations)
- [train.py](train.py): Unified training script for RNN / Transformer
- [inference.py](inference.py): Inference for RNN / Transformer / pretrained MT5
- [t5_finetune.py](t5_finetune.py): MT5 fine-tuning script
- [plot.py](plot.py): Plot loss/BLEU curves and ablation figures
- [dataset](dataset): Cleaned and retranslated Zh–En data (JSONL)
- [checkpoints](checkpoints): Trained model checkpoints
- [figures](figures): Generated figures for the report
- [mt5_translation_100k]: Checkpoints for T5 mini model finetune results 

## Data and Preprocessing

Raw data shows strong machine-translation artifacts and misalignment in `train_100k.jsonl` (e.g., obvious sentence shifts between indices 1471/1472). To obtain usable training data:

1. **Semantic cleaning**: Remove obviously misaligned Zh–En pairs via heuristics + manual checks.
2. **Retranslation**: Retranslate remaining pairs with Tencent Hunyuan MT, producing `train_100k_retranslated_hunyuan.jsonl`.
3. **Length filtering**: Apply `_validate_pair` (max_len, min_len, length ratio) and keep ~46k high-quality pairs.
4. **Evaluation max_len**: Use larger `max_len` (e.g., 512) at evaluation time to include more long sentences.

Tokenization:

- Chinese: Jieba
- English: SacreMoses or whitespace
- Word-level vocabularies plus SentencePiece models (`spm_zh`, `spm_en`)

RNN experiments use both word-level and SPM; Transformers mainly use word-level tokenization.

## Models

### RNN-based NMT

- Architecture: 2-layer unidirectional GRU encoder–decoder, embedding/hid_dim=256, dropout
- Attention: dot, general (multiplicative), additive (Bahdanau)
- Training: Teacher Forcing schedule with `tf_init`, `tf_decay`, `tf_min`
- Experiments:
  - Attention ablation: `rnn_dot_100k_bs128_heavy`, `rnn_gen_100k_bs128_heavy`, `rnn_add_100k_bs128_heavy`
  - Teacher Forcing vs. Free Running: `rnn_add_100k_bs128_heavy` vs. `rnn_add_100k_bs128_free`
  - Word vs. SentencePiece: `rnn_add_100k_bs128_heavy` vs. `rnn_sp_add_100k_heavy`, `rnn_sp_add_100k_maxlen256`

Key findings (validation BLEU, best epoch):

- Additive and general ≈ 7.8 BLEU; dot ≈ 6.9
- Higher `tf_min` (e.g., 0.3) stabilizes training and avoids BLEU collapse
- Under current hyperparameters, SPM RNN severely underperforms word-level (≈1.5–2.3 vs. 7.8)

### Transformer NMT (from scratch)

- Architecture: encoder–decoder; variants with
  - `d_model` ∈ {256, 512}
  - `nhead` ∈ {4, 8}
  - 4 or 6 layers per side
  - FFN sizes {512, 2048}
- Positional encodings:
  - Absolute sinusoidal vs. T5-style relative bias
- Normalization:
  - LayerNorm vs. RMSNorm

Key experiments:

- **pos/norm ablation** (256 dims):
  - `tf_abs_layer_256_100k_heavy` (abs + LayerNorm)
  - `tf_abs_rms_256_100k_heavy` (abs + RMSNorm)
  - `tf_rel_layer_256_100k_heavy` (rel + LayerNorm)
  - `tf_rel_rms_256_100k_heavy` (rel + RMSNorm)
- **Model scale & hyperparameters** (512 dims):
  - `tf_abs_layer_512_6l_100k`, `tf_abs_layer_512_6l_100k_bs64`, `tf_abs_layer_512_6l_100k_lr1e3`
- **Max length**:
  - `tf_abs_layer_256_100k_heavy` (max_len=128)
  - `tf_abs_layer_256_100k_maxlen256` (max_len=256)

Key findings:

- Relative position is slightly better than absolute (+0.2–0.4 BLEU), but not dominant
- LayerNorm vs. RMSNorm differences are small at this scale
- Larger models and batch sizes help, but LR too high (1e-3) harms stability
- Increasing max_len from 128 to 256 boosts validation BLEU from ~9.0 to ~11.4

### MT5 Fine-tuning

- Model: `google/mt5-small`
- Data: same cleaned 100k training set + retranslated val/test
- Input template: `translate Chinese to English: {zh}`
- Script: [t5_finetune.py](t5_finetune.py)

Current baseline (under non-fully-tuned hyperparameters):

- Validation BLEU ≈ 4.2–5.3
- Test BLEU ≈ 4.1
- Underperforms best Transformer (validation ≈ 11.4) but exceeds the weakest RNN variants

The report discusses likely reasons (limited steps, under-tuned LR/beam/length penalty, small data vs. MT5 pretraining domain).

## Training

General training entry point: [train.py](train.py).

Example commands (run from ``):

```bash
# Word-level RNN + additive attention
python train.py \
  --model rnn --attn additive \
  --emb_dim 256 --hid_dim 256 \
  --batch_size 128 --lr 3e-4 \
  --tf_init 0.9 --tf_decay 0.9 --tf_min 0.3 \
  --max_len 128 \
  --train_path dataset/train_100k_retranslated_hunyuan.jsonl \
  --valid_path dataset/val_retranslated.jsonl \
  --out_dir checkpoints/rnn_add_100k_bs128_heavy

# Base Transformer (256d, abs pos + LayerNorm)
python train.py \
  --model transformer --pos absolute --norm layer \
  --d_model 256 --nhead 4 --n_encoder 4 --n_decoder 4 --ff 512 \
  --batch_size 192 --lr 3e-4 \
  --scheduler cosine --warmup_steps 2000 \
  --max_len 128 \
  --train_path dataset/train_100k_retranslated_hunyuan.jsonl \
  --valid_path dataset/val_retranslated.jsonl \
  --out_dir checkpoints/tf_abs_layer_256_100k_heavy
```

MT5 fine-tuning example:

```bash
python t5_finetune.py \
  --model_name google/mt5-small \
  --train_path dataset/train_100k_retranslated_hunyuan.jsonl \
  --valid_path dataset/val_retranslated.jsonl \
  --output_dir mt5_translation_100k \
  --epochs 3 --batch_size 16 --lr 3e-4 --max_len 256
```

## Inference and Evaluation

RNN / Transformer inference is done via [inference.py](inference.py):

```bash
# Transformer, beam search on test set
python inference.py \
  --model transformer \
  --checkpoint checkpoints/tf_abs_layer_256_100k_maxlen256/best_transformer.pt \
  --data_path dataset/test_retranslated.jsonl \
  --strategy beam --beam 5 \
  --batch_size 64 --max_len 256 \
  --out_file outputs/tf_abs_layer256_beam5_test.txt
```

BLEU with sacrebleu:

```bash
# Extract references
ojq -r '.en' dataset/test_retranslated.jsonl > outputs/test_ref.txt

# Evaluate
sacrebleu outputs/test_ref.txt < outputs/tf_abs_layer256_beam5_test.txt
```

MT5 inference is done via the `pretrained` mode in `inference.py` (or a separate script with `AutoTokenizer`/`AutoModelForSeq2SeqLM`).

## Key Observations (High Level)

- **Data quality dominates**: Cleaning and retranslation are essential; raw 100k data is too noisy to train useful models.
- **RNN vs. Transformer**: Transformers clearly outperform RNNs on BLEU and long-sentence handling, at the cost of compute.
- **Attention & TF schedule**: Additive attention and non-zero `tf_min` give the best balance of stability and performance for RNNs.
- **Subwords are non-trivial**: Naïvely reusing word-level hyperparameters for SentencePiece RNN fails; subwords require re-tuning.
- **MT5 baseline**: Current fine-tuning underperforms the best scratch Transformer due to limited tuning, but highlights the potential of pretrained LMs.

For a detailed discussion, ablation figures, and full experimental results, see [report_en.typ](report_en.typ) (or the Chinese version [report.typ](report.typ)).
