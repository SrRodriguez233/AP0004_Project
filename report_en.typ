#import "@preview/scripst:1.1.1": *
#import "@preview/tablem:0.3.0": three-line-table

#show: scripst.with(
	template: "report",
	title: [Machine Translation between Chinese and English],
	author: ("250010127         王艺霖"),
	time: datetime.today().display(),
	font-size: 11pt,
	contents: true,
	content-depth: 2,
	matheq-depth: 2,
	counter-depth: 2,
	cb-counter-depth: 2,
	header: false,
	lang: "en",
)

= Introduction and Task Description

This project aims to build Chinese-to-English neural machine translation (NMT) systems and systematically compare three mainstream model families: an RNN-based NMT, a Transformer trained from scratch, and a fine-tuned pretrained MT5 model. Under a unified dataset and evaluation protocol, we analyze the differences between these models in terms of architecture, training efficiency, translation quality, and scalability.

The report is organized around the following core questions:

- What are the essential differences between sequential RNNs and fully parallel Transformers? RNNs rely on recurrent structures and process sequences step-by-step, which makes parallelization difficult; Transformers rely on self-attention, can model global dependencies, and support highly parallel computation.
- How do different attention mechanisms, positional encodings, normalization schemes, and hyperparameters affect performance? We compare three attention functions (dot/general/additive), absolute vs. relative positional encoding, LayerNorm vs. RMSNorm, and the impact of model scale, batch size, and learning rate.
- How should we trade off between training from scratch and fine-tuning a large pretrained model (such as MT5)? We compare their behavior under small and medium data settings, and discuss practical engineering cost and resource consumption.

All experiments are conducted on a unified Chinese–English parallel corpus (10k/100k training sets plus validation and test sets). BLEU is used as the primary evaluation metric. All models and pipelines are implemented from scratch, and the code and scripts for reproduction are available in the project repository #link("https://github.com/SrRodriguez233/AP0004_Project").


= Data and Preprocessing

== Datasets and Cleaning
The original dataset contains several JSONL files, including train_100k.jsonl, valid.jsonl and test.jsonl. A manual inspection shows that train_100k.jsonl contains a large amount of machine-translated text and severe semantic misalignment. For example, at index 1471 the English sentence is "Humanity has reached a critical moment...", while the corresponding Chinese sentence is "我们应该另辟蹊径，找到解决气候问题的新方法。" ("We should find new ways to solve climate problems."), and the correct Chinese translation seems to have been shifted to index 1472. Such mistranslations, omissions, and misalignments are common in the raw training set and strongly harm model learning.

To ensure training quality, we followed classmates’ suggestion and applied the following cleaning steps:

- *Semantic alignment checking*: We combined manual inspection and heuristic scripts to remove obviously misaligned sentence pairs, keeping only Chinese–English pairs that are semantically consistent.
- *Retranslation of the training set*: We retranslated the remaining training pairs using Tencent Hunyuan MT, producing train_100k_retranslated_hunyuan.jsonl. This significantly improves the accuracy and usability of the training data.
- *Length-based filtering*: We implemented a `_validate_pair` rule to filter sentence pairs by length (max_len, min_len, and length ratio). After filtering, about 46k high-quality pairs are retained, which avoids extremely long or abnormal sentences that may destabilize training.
- *Evaluation handling*: During testing, we increased max_len (e.g., to 512) so that more long sentences are included in evaluation, making the test results more comprehensive and representative.

== Tokenization and Vocabularies
1. *Chinese tokenization*: We use Jieba to segment Chinese sentences, balancing speed and accuracy.
2. *English tokenization*: We use SacreMoses or whitespace tokenization to standardize English text.
3. *Word and subword vocabularies*: We build word-level vocabularies with a frequency threshold, and SentencePiece subword models (spm_zh / spm_en), with agreed special token IDs for pad/bos/eos/unk. In experiments, RNN models are trained with both word-level and subword tokenization; Transformers primarily use word-level tokenization.

These cleaning and retranslation steps substantially improve data quality, enabling stabler convergence and more reasonable BLEU scores. Implementation details of preprocessing are in data_utils.py, and all subsequent experiments are based on the cleaned dataset.

= RNN-based NMT Experiments
This section first introduces the overall encoder–decoder RNN NMT architecture, then presents three lines of comparison experiments: attention mechanisms, training strategies, and tokenization schemes. The code is mainly in rnn_model.py and train.py.

== Word-level Tokenization with Three Attention Mechanisms

=== Model Architecture
We use a two-layer unidirectional GRU encoder and decoder with embedding/hid_dim=256 and dropout, built on a word-level vocabulary. The encoder encodes the Chinese source sentence into a sequence of hidden states; the decoder predicts each target token step-by-step based on the previous output, current hidden state, and attention context. The loss is cross entropy with padding positions masked out, the optimizer is Adam, and a typical configuration is batch_size=128, learning rate 3e-4, training for multiple epochs until validation BLEU converges.

=== Attention Mechanisms
We implement dot-product, general (multiplicative), and additive (Bahdanau/concat) attention. Dot and general attention are computationally cheaper and are suitable when the hidden dimension is larger; additive attention uses a small feed-forward layer to compute alignment scores, with more parameters but in our setting provides the most stable convergence and better alignment on long sentences.

=== Training Strategy
We use Teacher Forcing with tf_init=0.9, tf_decay, and tf_min=0.5 to ensure stable training. Concretely, the ground-truth previous token is fed into the decoder with probability TF, and we gradually reduce TF according to tf_init/tf_decay/tf_min so that in later stages the model is exposed to its own predictions, mitigating exposure bias.

=== Checkpoints

- checkpoints/rnn_dot_100k_bs128_heavy/best_rnn.pt
- checkpoints/rnn_gen_100k_bs128_heavy/best_rnn.pt
- checkpoints/rnn_add_100k_bs128_heavy/best_rnn.pt

=== Training Curves

#figure(
	image("figures/rnn_nmt_attn_loss.png"),
	caption: "Training loss of RNN with different attention mechanisms",
)

All three losses decrease smoothly; additive attention drops the fastest, dot attention is slightly slower, and general lies in between.

#figure(
	image("figures/rnn_nmt_attn_bleu.png"),
	caption: "Validation BLEU of RNN with different attention mechanisms",
)

The BLEU curves all show noticeable oscillations: additive has fluctuations but an overall upward trend; dot tends to drift downward; general is more stable and lies between them.

=== Results
From the log_rnn.csv files in the three checkpoint folders we read approximate peak validation BLEU: general ≈ 7.8, additive ≈ 7.8, dot ≈ 6.9. In other words, general and additive have similar best BLEU and both clearly outperform dot. Their main difference lies in convergence smoothness and robustness on long sentences.

#figure(
	three-line-table()[
	|Checkpoint directory           |Validation BLEU (best)|
	|---                            |---                   |
	|rnn_gen_100k_bs128_heavy      |7.8                   |
	|rnn_add_100k_bs128_heavy      |7.8                   |
	|rnn_dot_100k_bs128_heavy      |6.9                   |
	],
	caption: "Best validation BLEU of RNN with different attention mechanisms",
)


Inspecting the training logs further, we find that general attention quickly reaches high BLEU in the first few epochs but later behaves roughly like the average of additive and dot; additive attention, although peaking at a similar BLEU as general, has smoother loss/BLEU curves, with fewer spikes, and is less sensitive to gradient clipping and learning rate choices. Considering both performance and stability, we use additive as the default attention for later RNN experiments (with general as a secondary option) and no longer use dot.

== Teacher Forcing vs. Free Running

=== Checkpoints

- checkpoints/rnn_add_100k_bs128_heavy/best_rnn.pt (tf_init=0.9, tf_decay=0.9, tf_min=0.3, stable)
- checkpoints/rnn_add_100k_bs128_free/best_rnn.pt (tf_init=0.49, tf_decay=0.7, tf_min=0, unstable BLEU)

=== Training Curves

#figure(
	image("figures/rnn_teacher_free_attn_loss.png"),
	caption: "Training loss of RNN under Teacher Forcing vs. Free Running",
)

The loss curves show that the model with tf_min=0.3 descends faster, while the tf_min=0 model descends more slowly.

#figure(
	image("figures/rnn_teacher_free_attn_bleu.png"),
	caption: "Validation BLEU of RNN under Teacher Forcing vs. Free Running",
)

The BLEU curves indicate that with tf_min=0.3, BLEU keeps improving with moderate fluctuations and stays above 7 in later epochs; with tf_min=0, BLEU quickly jumps to 6.6 in the first two epochs, but then gradually drops back to 5–6 with large oscillations.

=== Results

#figure(
three-line-table()[
	|Checkpoint directory           |tf_min|Validation BLEU (best)|
	|---                            |---   |---                    |
	|rnn_add_100k_bs128_heavy      |0.30  |7.8                    |
	|rnn_add_100k_bs128_free       |0.00  |6.6                    |
],
caption: "Best validation BLEU of RNN under different Teacher Forcing schedules",
)


=== Phenomena and Analysis

- A higher tf_min (0.3) effectively suppresses exposure bias: validation BLEU fluctuates within 6.5–7.8 and stays above 7 in later epochs.
- With tf_min=0, the Teacher Forcing ratio drops to near zero very quickly. BLEU peaks at 6.62 in epoch 2, but as TF decays further (tf≈0.24 → 0.0000x), BLEU gradually falls back to around 5–6. Although training loss continues decreasing, validation BLEU deteriorates, a typical combination of overfitting and exposure bias. Generated sequences are more prone to repetition and prematurely outputting <eos>.
- Conclusion: keeping a non-zero tf_min (e.g., 0.3) is a safer strategy. If we want to reduce TF, it is better to slow down the decay or raise tf_min, instead of decaying to zero.

== RNN with SentencePiece Subword Tokenization

=== Checkpoints

- checkpoints/rnn_sp_add_100k_heavy/best_rnn.pt (SentencePiece subwords, additive attention)
- checkpoints/rnn_add_100k_bs128_heavy/best_rnn.pt (word-level, additive attention)
- checkpoints/rnn_sp_add_100k_maxlen256/best_rnn.pt (SentencePiece subwords, additive attention, max_len=256)

=== Training Curves

#figure(
	image("figures/rnn_sp_attn_loss.png"),
	caption: "Training loss of RNN with SentencePiece subword tokenization",
)

#figure(
	image("figures/rnn_sp_attn_bleu.png"),
	caption: "Validation BLEU of RNN with SentencePiece subword tokenization",
)

=== Results (best validation BLEU)

#figure(
	three-line-table()[
	|Checkpoint directory           |Tokenization  |max_len|Validation BLEU (best)|
	|---                            |---           |---    |---                    |
	|rnn_add_100k_bs128_heavy      |word          |128    |7.8                    |
	|rnn_sp_add_100k_heavy         |SentencePiece |128    |1.54                   |
	|rnn_sp_add_100k_maxlen256     |SentencePiece |256    |2.31                   |
],
	caption: "Best validation BLEU of RNN: word-level vs. SentencePiece subwords",
)

=== Phenomena and Analysis

- Under the same network and number of epochs, the word-level model reaches validation BLEU≈7.8, while the subword models only reach ≈1.5 (max_len=128) or ≈2.3 (max_len=256). They are far from a usable level.
- Likely reasons:
	- Subword sequences are much longer; with max_len=128 many long sentences are truncated, so the model cannot learn complete alignments.
	- Training hyperparameters (learning rate, number of epochs, Teacher Forcing schedule) are copied from the word-level setting and are not retuned for longer sequences and larger effective time steps, which slows convergence.
	- RNNs are more sensitive to long sequences than Transformers. Long sequences plus subwords increase the risk of gradient vanishing/explosion; even with gradient clipping, the model needs more training to stabilize.

- Increasing max_len to 256 leads to some improvement but still falls short of the word-level model. To really benefit from subword modeling and better handling of OOV words, we likely need a larger max_len, more training epochs, and adjusted learning rate/gradient clipping.

=== Conclusion

Under the current hyperparameters and max_len, the subword RNN fails to outperform the word-level model. Given limited time and compute, we therefore choose the better-converged word-level model as the main RNN result, while honestly reporting and analyzing the negative subword results.

= Transformer-based NMT Experiments

This section briefly introduces our custom Transformer: an encoder–decoder architecture that supports sinusoidal absolute positional encoding and T5-style relative position bias; normalization can be either LayerNorm or RMSNorm; boolean masks are used in multi-head attention to avoid CUDA assertions, and relative position indices are clipped to avoid out-of-bounds access.

== Base-scale Transformer (256 dims) — pos/norm Ablation

=== Model Architecture

We use d_model=256, nhead=4, 4 encoder and 4 decoder layers, FFN=512, dropout, and word-level tokenization. Both training and validation use the 100k dataset variant.

=== Checkpoints

- checkpoints/tf_abs_layer_256_100k_heavy/best_transformer.pt (absolute position + LayerNorm)
- checkpoints/tf_abs_rms_256_100k_heavy/best_transformer.pt (absolute position + RMSNorm)
- checkpoints/tf_rel_layer_256_100k_heavy/best_transformer.pt (relative position + LayerNorm)
- checkpoints/tf_rel_rms_256_100k_heavy/best_transformer.pt (relative position + RMSNorm)

=== Training Curves

#figure(
	image("figures/tf_pos_norm_loss.png"),
	caption: "Training loss of Transformer with different positional encodings and normalization",
)

#figure(
	image("figures/tf_pos_norm_bleu.png"),
	caption: "Validation BLEU of Transformer with different positional encodings and normalization",
)

=== Results (best validation BLEU)

#figure(
	three-line-table()[
	|Checkpoint directory                |pos |norm      |Validation BLEU (best)|
	|---                                 |--- |---       |---                    |
	|tf_abs_layer_256_100k_heavy        |abs |LayerNorm |9.04                   |
	|tf_abs_rms_256_100k_heavy          |abs |RMSNorm   |8.49                   |
	|tf_rel_layer_256_100k_heavy        |rel |LayerNorm |8.88                   |
	|tf_rel_rms_256_100k_heavy          |rel |RMSNorm   |8.66                   |
],
	caption: "Best validation BLEU of Transformer with different pos/norm combinations",
)

=== Phenomena and Analysis

- Relative position is slightly better than absolute (≈+0.2–0.4 BLEU), but the gap is small. At current data scale, relative distance modeling helps, but is not decisive.
- The difference between LayerNorm and RMSNorm is also small; RMSNorm does not clearly surpass LayerNorm. The training curves are smooth in all cases, indicating that the base configuration is reasonably stable.
- All four setups converge to BLEU in the 8–9 range. To go beyond, we need larger models, longer max_len, or more aggressive hyperparameter tuning.

== Larger Transformer and Hyperparameter Sensitivity (512 dims)

=== Model Architecture

We use d_model=512, nhead=8, 6 encoder and 6 decoder layers, FFN=2048, with other settings the same as for the 256-dim group.

=== Checkpoints

- checkpoints/tf_abs_layer_512_6l_100k/best_transformer.pt (batch=128, lr=5e-4)
- checkpoints/tf_abs_layer_512_6l_100k_bs64/best_transformer.pt (batch=64, lr=5e-4)
- checkpoints/tf_abs_layer_512_6l_100k_lr1e3/best_transformer.pt (batch=128, lr=1e-3)

=== Training Curves

#figure(
	image("figures/tf_large_loss.png"),
	caption: "Training loss of larger Transformer models",
)

#figure(
	image("figures/tf_large_bleu.png"),
	caption: "Validation BLEU of larger Transformer models",
)

=== Results (best validation BLEU)

#figure(
	three-line-table()[
	|Checkpoint directory                |d_model|batch|lr   |Validation BLEU (best)|
	|---                                 |---    |---  |---  |---                    |
	|tf_abs_layer_512_6l_100k           |512    |128  |5e-4 |8.89                   |
	|tf_abs_layer_512_6l_100k_bs64      |512    |64   |5e-4 |8.52                   |
	|tf_abs_layer_512_6l_100k_lr1e3     |512    |128  |1e-3 |7.83                   |
],
	caption: "Best validation BLEU of larger Transformers with different batch/lr",
)

=== Phenomena and Analysis

- Increasing model size plus suitable batch/lr leads to BLEU≈8.5–8.9; reducing the batch size (to 64) slightly degrades BLEU (8.52), indicating that larger batch sizes help stabilize gradients.
- Raising the learning rate to 1e-3 reduces BLEU to ≈7.8 and increases oscillations, suggesting that we need warmup or a lower initial LR and stronger regularization.
- Larger models substantially increase training time and memory consumption, so gradient clipping and proper LR scheduling are important to avoid overfitting and numerical instability.

== Increasing Maximum Sentence Length to 256

=== Checkpoints

- checkpoints/tf_abs_layer_256_100k_maxlen256/best_transformer.pt (max_len=256, otherwise same as base 256-dim model)
- checkpoints/tf_abs_layer_256_100k_heavy/best_transformer.pt (max_len=128, baseline)

=== Training Curves

#figure(
	image("figures/tf_maxlen256_loss.png"),
	caption: "Training loss of Transformer with max_len=256",
)

#figure(
	image("figures/tf_maxlen256_bleu.png"),
	caption: "Validation BLEU of Transformer with max_len=256",
)

=== Results (best validation BLEU)

#figure(
	three-line-table()[
	|Checkpoint directory                |max_len|Validation BLEU (best)|
	|---                                 |---    |---                    |
	|tf_abs_layer_256_100k_heavy        |128    |9.04                   |
	|tf_abs_layer_256_100k_maxlen256    |256    |11.43                  |
],
	caption: "Effect of max_len on best validation BLEU for Transformer",
)

=== Phenomena and Analysis

- Increasing max_len to 256 yields a significant jump in validation BLEU to 11.43, indicating that longer input sequences allow the model to capture more context, especially beneficial for long-sentence translation.
- The training curves show that the max_len=256 model converges faster in early epochs and produces more stable BLEU later, supporting the idea that longer sentences help the model learn richer structures.
- The tradeoff is increased training time and memory usage, so max_len must be chosen based on resource constraints and performance requirements.

= Fine-tuning a Pretrained LM (MT5)

This section uses HuggingFace’s google/mt5-small for Chinese→English MT, and compares it with our Transformers trained from scratch.

== Experimental Setup

- Model: google/mt5-small (encoder–decoder, built-in SentencePiece)
- Data: train_100k_retranslated_hunyuan.jsonl (train), val_retranslated.jsonl (validation), test_retranslated.jsonl (test)
- Input template: `translate Chinese to English: {zh}`
- Tokenization: MT5’s built-in SentencePiece
- Training script: t5_finetune.py
- Key hyperparameters: learning_rate=3e-4, beam=5 at decoding
- Output directory: mt5_translation_100k/ (best checkpoint around step 15000)

== Results

#figure(
	three-line-table()[
	|Model                      |Validation BLEU (best)|Test BLEU|
	|---                        |---                   |---      |
	|mt5_finetune 15105 steps   |5.3                   |4.1      |
	|mt5_finetune 15000 steps   |5.3                   |4.1      |
	|mt5_finetune 10000 steps   |5.0                   |--       |
	|mt5_finetune 5000 steps    |4.2                   |--       |
],
	caption: "Validation and test BLEU of MT5 fine-tuning",
)

== Phenomena and Analysis

- On the 100k dataset, MT5 fine-tuning converges relatively quickly: validation BLEU fluctuates between 4.2 and 5.3, suggesting that more training or better hyperparameter tuning might be beneficial.
- Test BLEU is around 4.1, which is lower than our Transformers trained from scratch. This may be due to inadequate fine-tuning steps or suboptimal hyperparameters.
- MT5 requires significantly more training time and GPU memory than our custom Transformers, but in principle has stronger language understanding and generation ability and is more suitable for larger and more complex tasks.


= Overall Comparison and Analysis

== Training Efficiency

RNNs train faster and use less memory, making them attractive for resource-constrained settings.

Transformers train slower and require more GPU memory but converge faster and achieve higher BLEU.

Larger Transformers (d_model=512, 6 layers) need several hours to train and consume much more memory.

MT5 is expected to outperform scratch Transformers in principle, but current validation/test evaluations are limited and under-optimized; its per-step cost and memory usage are significantly higher than our custom Transformers, so we must weigh performance against hardware constraints.

== Translation Performance (BLEU)

#figure(
	three-line-table()[
	|Model                             |Validation BLEU (best)|Remark                                   |
	|---                               |---                   |---                                      |
	|rnn_add_100k_bs128_heavy         |7.8                   |word-level, tf_min=0.30                 |
	|rnn_add_100k_bs128_free          |6.6                   |word-level, tf_min=0                    |
	|rnn_sp_add_100k_maxlen256        |2.31                  |SentencePiece, max_len=256              |
	|tf_abs_layer_256_100k_heavy      |9.04                  |abs+LayerNorm, d_model=256              |
	|tf_rel_rms_256_100k_heavy        |8.66                  |rel+RMSNorm, d_model=256                |
	|tf_abs_layer_512_6l_100k         |8.89                  |large model, batch=128, lr=5e-4         |
	|tf_abs_layer_256_100k_maxlen256  |11.43                 |max_len=256, d_model=256                |
	|mt5_finetune_15000 steps         |5.3                   |mini MT5 fine-tuning baseline           |
	],
	caption: "Best validation BLEU of different models (test BLEU omitted or reported separately)",
)

Overall, the base Transformer (256 dims) improves upon the word-level RNN by about 1–2 BLEU; larger models and different pos/norm configurations lie in the 8–9 BLEU range; RNN with subwords performs worst in the current setting. In this round of experiments, MT5 underperforms the best Transformer (though still better than RNN), likely due to insufficient fine-tuning and non-optimal hyperparameters; more work is needed to fully exploit its potential.

== Decoding Strategies: Greedy vs. Beam Search

In previous experiments, we generated both greedy and beam=5 translations for RNN and Transformer models, and evaluated them on the same validation/test sets using sacrebleu. This provides a unified view of how decoding strategies affect BLEU and latency.

In general, beam search keeps multiple candidate paths at each time step and usually improves validation/test BLEU by about 1–2 points, especially for long sentences or those with many paraphrases, where beam can select more complete and fluent translations. However, the computational cost grows roughly linearly with beam size, so average decoding time and GPU usage increase significantly.

In our experiments, RNNs benefit relatively more from beam (each step is cheap), while Transformers and MT5 become the main source of inference latency when beam=5. In practice, we must balance BLEU gain and real-time requirements: for offline high-quality translation, a larger beam is acceptable; for online interactive systems, greedy or small beam sizes (e.g., 3–5) are more appropriate.

== Generalization and Scalability

- *Long sentences*: Transformers with larger max_len (e.g., 256 or 512) handle long sentences better, whereas RNNs tend to lose long-range information.
- *Model scaling*: Transformers scale well with depth and width; RNNs are more constrained by recurrence.
- *Low-resource settings*: Transformers with subwords handle low-resource scenarios more gracefully; RNNs degrade faster on the 10k training set. Subword modeling and capacity/regularization design are important to maintain performance.

With SentencePiece and careful regularization, Transformers can maintain relatively stable BLEU even on the 10k dataset, while RNNs drop more sharply, highlighting the importance of model capacity and regularization.

== Practical Trade-offs

RNNs are small, fast at inference, and easy to implement, suitable for environments with tight resource budgets.

Transformers offer higher performance but are more sensitive to hyperparameters and require more tuning, making them suitable for larger datasets and higher accuracy requirements.

Regarding decoding, beam search usually gives modest BLEU gains over greedy but at the cost of roughly linear increases in latency. In this project, we ultimately adopt beam=5 as a compromise for RNN/Transformer/MT5 in most evaluations, achieving better fluency at acceptable delay.

== Implementation and Reproducibility

To facilitate reproduction of our results, we briefly describe the code structure and typical commands.

- *Code structure*
	- train.py: unified training script. The flag --model {rnn, transformer} and additional arguments (--attn, --d_model, --nhead, --pos, --norm, --max_len, etc.) control the network architecture and optimization settings.
	- inference.py: inference and generation script for RNN/Transformer, with --strategy {greedy, beam} and --beam controlling decoding.
	- t5_finetune.py: MT5 fine-tuning script based on HuggingFace Transformers, using Seq2SeqTrainer and Seq2SeqTrainingArguments.
	- data_utils.py: data loading, cleaning and tokenization, including the `_validate_pair` filter and `vocabulary/SentencePiece` construction.
	- plot.py: reads log_rnn.csv/log_transformer.csv from each checkpoint directory and generates unified loss/BLEU plots in figures/.

- *Typical training commands*
	- Word-level RNN with additive attention (main RNN model):
		- python train.py --model rnn --attn additive --emb_dim 256 --hid_dim 256 --batch_size 128 --lr 3e-4 --tf_init 0.9 --tf_decay 0.9 --tf_min 0.3 --max_len 128 --train_path dataset/train_100k_retranslated_hunyuan.jsonl --valid_path dataset/val_retranslated.jsonl --out_dir checkpoints/rnn_add_100k_bs128_heavy
	- Base Transformer (256 dims, absolute pos + LayerNorm):
		- python train.py --model transformer --pos absolute --norm layer --d_model 256 --nhead 4 --n_encoder 4 --n_decoder 4 --ff 512 --batch_size 192 --lr 3e-4 --scheduler cosine --warmup_steps 2000 --max_len 128 --train_path dataset/train_100k_retranslated_hunyuan.jsonl --valid_path dataset/val_retranslated.jsonl --out_dir checkpoints/tf_abs_layer_256_100k_heavy
	- Large Transformer (512 dims, 6 encoder/decoder layers):
		- python train.py --model transformer --pos absolute --norm layer --d_model 512 --nhead 8 --n_encoder 6 --n_decoder 6 --ff 2048 --batch_size 128 --lr 5e-4 --scheduler cosine --warmup_steps 2000 --max_len 128 --train_path dataset/train_100k_retranslated_hunyuan.jsonl --valid_path dataset/val_retranslated.jsonl --out_dir checkpoints/tf_abs_layer_512_6l_100k
	- MT5 fine-tuning (simplified example):
		- python t5_finetune.py --model_name google/mt5-small --train_path dataset/train_100k_retranslated_hunyuan.jsonl --valid_path dataset/val_retranslated.jsonl --output_dir mt5_translation_100k --epochs 3 --batch_size 16 --lr 3e-4 --max_len 256

- *Inference and evaluation*
	- RNN/Transformer inference via inference.py, e.g.:
		- python inference.py --model transformer --checkpoint checkpoints/tf_abs_layer_256_100k_heavy/best_transformer.pt --data_path dataset/test_retranslated.jsonl --strategy beam --beam 5 --batch_size 64 --max_len 256 --out_file outputs/tf_abs_layer256_beam5_test.txt
	- Plotting and BLEU evaluation:
		- Use `plot.py` to read `checkpoints/*/log_*.csv` and save loss/BLEU curves to `figures/`.
		- Use `sacrebleu` on `outputs/*_test.txt` against reference test files to compute test BLEU and fill in report tables.

All commands and scripts have been tested in the project root and can reproduce the main experiments on a Linux + Python + PyTorch + Transformers stack.

== Further Discussion of Data and Model Behavior

In previous sections we already reported BLEU under a unified setup for RNN, Transformer, and MT5. Here we reflect more deeply on some "abnormal" phenomena from three angles: data quality, tokenization strategy, and pretrained model behavior.

=== Training Data Quality and Distribution Shift

Although we cleaned and retranslated train_100k.jsonl and ended up with about 46k relatively clean pairs, this has two consequences:

- The cleaned training set is much higher quality than the original 100k, which helps stable convergence and avoids learning noisy alignments.
- However, the effective sample size is nearly halved, and the validation/test sets are still the original retranslated splits. Their style and domain are not perfectly aligned with the cleaned training set, leading to some distribution mismatch.

In practice we observe:

- Both RNN and Transformer achieve reasonably stable BLEU on training and validation (RNN around 7–8, Transformer around 9+), but the absolute numbers are not very high.
- On the test set, especially for long or syntactically complex sentences, models still tend to oversimplify, drop information, or sound unnatural.

Overall, this project is closer to training on a "noisy and small" realistic dataset, rather than an ideal large-scale clean corpus. We intentionally highlight this in the preprocessing section to avoid over-interpreting BLEU as a measure of absolute translation quality.

=== Why the Subword RNN Experiments "Fail"

As shown, RNN + SentencePiece performs poorly under current hyperparameters:

- Word-level RNN reaches validation BLEU≈7.8.
- Subword RNN is only ≈1.5 (max_len=128) or ≈2.3 (max_len=256), far worse than the word-level model.

Combining logs and training configs, the main reasons are:

- *Longer sequences*: Subword tokenization roughly doubles or triples sequence length. With max_len=128, many sentences are truncated in both training and evaluation, preventing the model from seeing full context.
- *Hyperparameters copied from word-level*: LR, epochs, and Teacher Forcing schedule are directly inherited from the word-level setting, without retuning for longer sequences. This makes optimization harder and slower.
- *RNN sensitivity to length*: Compared to Transformers, GRUs are more fragile on long sequences; long subword sequences push them toward gradient issues even with clipping.

Thus, these experiments are a useful "negative example": naïvely porting word-level hyperparameters to subword RNN leads to poor results. This does not mean subwords are useless, but that they require dedicated tuning (longer max_len, more epochs, adjusted LR, better clipping).

=== Why MT5 Underperforms Scratch Transformers Here

In this project, MT5 fine-tuning achieves validation BLEU≈4–5 and test BLEU≈4.1, significantly lower than our best Transformer (validation BLEU≈11.4). This contradicts the common belief that pretrained LMs outperform small scratch models, and likely stems from several factors:

- *Insufficient fine-tuning*: On 100k data, the number of effective epochs for MT5 is small. Multiple checkpoints were saved, but trainer_state lacks systematic eval_bleu tracking, suggesting that earlier training runs did not follow a strict "evaluate regularly and select best BLEU" strategy.
- *Hyperparameters and decoding not fully tuned*:
	- Learning rate, batch size, and gradient accumulation are chosen by intuition from small-model training rather than via a careful grid search for MT5.
	- Generation hyperparameters at evaluation time (beam size, length penalty, max_length, etc.) are also under-tuned, which can cap translation quality.
- *Task/domain mismatch and low data regime*: 46k effective training pairs is tiny compared to the massive multilingual corpora used in MT5 pretraining. Our retranslated + cleaned data also has a relatively narrow style, which may not fully leverage MT5’s broad pretraining.

In short, current MT5 results reflect a preliminary baseline rather than an upper bound. We choose to present these numbers honestly and discuss likely reasons and improvements (more steps, better lr/beam/length penalty tuning, more robust evaluation strategy), which both shows the potential of pretrained models and acknowledges the limits of this project.

= Conclusion and Outlook

This project implements and compares RNN and Transformer NMT models, covering attention mechanisms, tokenization choices, positional encodings, normalization, and hyperparameter sensitivity. The experiments show that data quality and preprocessing have a large impact on performance; Transformers excel on longer sentences and larger data, while RNNs are simple and efficient but limited in capacity.

On the engineering side, we build the entire pipeline from scratch: data handling, RNN/Transformer architectures, Teacher Forcing scheduling, greedy/beam decoding, and MT5 fine-tuning. Through systematic experiments we learn that model choice is only the first step; data cleaning, tokenization, training hyperparameters, and decoding strategies are equally important for final BLEU.

Future work may explore stronger tokenizers, larger pretrained models, and hybrid RNN–Transformer architectures, as well as more systematic hyperparameter searches for MT5 to fully realize its potential.
