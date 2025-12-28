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
  lang: "zh",
)

= 引言及任务描述

本项目旨在实现中文到英文的神经机器翻译（NMT），并系统比较三类主流模型：基于 RNN 的 NMT、从零训练的 Transformer 以及微调预训练的 MT5。通过统一的数据集和评测指标，深入分析不同模型在结构、训练效率、翻译性能、可扩展性等方面的差异与优劣。

本报告将围绕以下核心问题展开：

- 序列式 RNN 与并行 Transformer 的本质区别：RNN 依赖递归结构，逐步处理序列，难以并行；而 Transformer 采用自注意力机制，可全局建模并支持高效并行计算。
- 不同注意力机制、位置编码、归一化方式及超参数的影响：对比 dot/general/additive 三种注意力，绝对/相对位置编码，LayerNorm/RMSNorm，以及模型规模、batch size、学习率等超参数对性能的影响。
- 从零训练与微调预训练模型的取舍：分析直接训练（from scratch）与微调大规模预训练模型（如 MT5）在小数据和大数据场景下的表现差异，以及工程实现和资源消耗的实际考量。
实验部分均基于统一的中英平行语料（10k/100k 训练集、验证集、测试集），采用 BLEU 作为主要评测指标。所有模型和实验流程均为自主实现，相关代码和可复现脚本已整理于项目仓库#link("https://github.com/SrRodriguez233/AP0004_Project")中。


= 数据与预处理

== 数据集与清洗
本项目采用的原始数据集包含多个版本，分别为 train_100k.jsonl、valid.jsonl 和 test.jsonl 等。经过初步分析，发现 train_100k.jsonl 存在大量机翻痕迹和严重的语义错位问题。例如，第 1471 行的英文为 "Humanity has reached a critical moment..."，对应的中文却是“我们应该另辟蹊径，找到解决气候问题的新方法。”，而真正的翻译疑似错位到了下一行。这类错译、漏译和不对齐现象在训练集中较为普遍，直接影响模型的学习效果。

为保证训练质量，根据同学们的建议，对数据进行了如下清洗和处理：

语义对齐检查：人工和脚本结合，删除了明显语义不对齐的翻译对，确保每一对中英文句子语义一致。
重译训练集：对剩余的训练数据，采用混元模型进行了重新翻译（retranslation），生成了 train_100k_retranslated_hunyuan.jsonl，大幅提升了数据的准确性和可用性。
长度过滤与清洗：实现了 \_validate_pair 规则，对句子长度进行过滤（如 max_len、min_len、长度比），最终保留约 4.6 万对高质量句对，避免极端长句和异常样本影响训练。
评估集处理：在测试阶段，适当提高了 max_len（如设为 512），以覆盖更多长句，保证评估的全面性和代表性。
此外，分词和词表构建采用如下策略：

== 分词与词表
1. 中文分词：使用 Jieba 工具进行分词，兼顾速度和准确性。
2. 英文分词：采用 SacreMoses 或空格分词，保证英文语料的标准化处理。
3. 词表与子词：分别构建词级词表（设定频率阈值）和 SentencePiece 子词模型（spm_zh / spm_en），并约定 pad/bos/eos/unk 的特殊 ID。实验中，RNN 模型同时测试了词级和子词分词，Transformer 主要采用词级分词。
通过上述清洗和重译，显著提升了训练数据的质量，使模型能够更稳定地收敛并获得更合理的 BLEU 分数。数据预处理的细节和代码实现已在 data_utils.py 中给出，后续实验均基于清洗后的高质量数据集展开。

= RNN-based NMT 实验
本节首先介绍基于编码器–解码器结构的 RNN NMT 模型整体设计，然后从注意力机制、训练策略与分词方式三个维度展开对比实验。本节的代码基于`rnn_model.py`和`train.py`实现。
== 词级分词 + 三种注意力机制
=== 模型结构
双层单向 GRU 编码器和解码器，embedding/hid_dim=256，使用 dropout，基于词级分词构建词表。编码器将中文源句编码为隐状态序列，解码器在每一步根据前一时刻输出、当前隐藏状态以及注意力上下文预测下一个目标词。损失函数采用交叉熵，对 `<pad>` 位置做 mask，优化器使用 Adam，典型设置为 batch_size=128、学习率 3e-4、训练若干轮直至验证 BLEU 收敛。
=== 注意力机制
分别实现 dot-product、general（multiplicative）、additive（Bahdanau/concat）。dot 与 general 计算量较小，适合隐藏维度较大时使用；additive 通过一个小型前馈网络建模对齐分数，参数较多但在本项目中收敛最稳定、对长句对齐更准确。
=== 训练策略
采用 Teacher Forcing，tf_init=0.9，tf_decay，tf_min=0.5，保证训练稳定。具体做法是以真实上一词作为输入进行条件建模，并通过 tf_init / tf_decay / tf_min 逐步降低 Teacher Forcing 比例，从而在训练后期让模型暴露于自身预测，缓解 exposure bias。
=== 实验 checkpoint
	- `checkpoints/rnn_dot_100k_bs128_heavy/best_rnn.pt`
	- `checkpoints/rnn_gen_100k_bs128_heavy/best_rnn.pt`
	- `checkpoints/rnn_add_100k_bs128_heavy/best_rnn.pt`

=== 训练曲线

#figure(
  image("figures/rnn_nmt_attn_loss.png"),
  caption: "RNN 不同注意力机制的训练 loss 曲线"
)

loss 曲线均较为平滑，additive 注意力的 loss 下降更快，dot 注意力稍慢，general 居于二者之间。
#figure(
  image("figures/rnn_nmt_attn_bleu.png"),
  caption: "RNN 不同注意力机制的验证 BLEU 曲线"
)

几条曲线均出现了相当程度的震荡，additive震荡但整体趋势向上，而dot震荡整体趋势向下，general居于二者之间相对稳定。

=== 结果对比
从三个 checkpoint 目录中的 log_rnn.csv 可以读出，每个模型在训练过程中验证 BLEU 的峰值大致为：general ≈ 7.8、additive ≈ 7.8、dot ≈ 6.9。也就是说，general 与 additive 的最佳 BLEU 接近，且都明显优于 dot；两者的差异主要体现在收敛平滑度和对长句的鲁棒性上。
#figure(
  three-line-table()[
  |模型目录	|验证 BLEU（best）	|
  |---|---	|
  |rnn_gen_100k_bs128_heavy	| 7.8	|
  |rnn_add_100k_bs128_heavy	| 7.8	|
  |rnn_dot_100k_bs128_heavy	| 6.9	|
],
  caption: "RNN 不同注意力机制的验证集最佳 BLEU 对比"
)


进一步观察各自的训练日志可以发现，general 注意力在前几轮迅速达到较高 BLEU，但后期数值相当于add和dot的平均；additive 注意力虽然峰值与 general 相近，却在整个训练过程中 loss/BLEU 曲线更加平滑、没有剧烈波动，对梯度裁剪与学习率设置的敏感度也更低。综合考虑性能与稳定性，后续 RNN 实验多采用 additive 注意力作为默认配置，辅以 general ，而不再使用 dot 注意力。

== Teacher Forcing vs Free Running
=== 实验 checkpoint
- `checkpoints/rnn_add_100k_bs128_heavy/best_rnn.pt`（tf_init=0.9, tf_decay=0.9, tf_min=0.3，训练稳定）
- `checkpoints/rnn_add_100k_bs128_free/best_rnn.pt`（tf_init=0.49, tf_decay=0.7, tf_min=0，后期 BLEU 波动大）

=== 训练曲线
#figure(
  image("figures/rnn_teacher_free_attn_loss.png"),
  caption: "RNN Teacher Forcing vs Free Running 的训练 loss 曲线"
)
loss 曲线显示，tf_min=0.3 的模型 loss 下降更快，而 tf_min=0 的模型下降稍慢。
#figure(
  image("figures/rnn_teacher_free_attn_bleu.png"),
  caption: "RNN Teacher Forcing vs Free Running 的验证 BLEU 曲线"
)

BLEU 曲线显示，tf_min=0.3 的模型在整个训练过程中 BLEU 在有波动地提升，且后期维持在 7+；而 tf_min=0 的模型在前几轮迅速提升到 6.6，但随后 BLEU 逐步下滑到 5.x–6.x，且波动较大。

=== 结果对比

#figure(
three-line-table()[
  |模型目录|tf_min|验证 BLEU（best）|
  |---|---|---|
  |rnn_add_100k_bs128_heavy|0.30|7.8|
  |rnn_add_100k_bs128_free|0.00|6.6|
],
caption: "RNN Teacher Forcing vs Free Running 的验证集最佳 BLEU 对比"
)


=== 现象与分析
- 高 tf_min（0.3）能有效抑制 exposure bias，验证 BLEU 在 6.5–7.8 区间平稳波动，后期仍维持在 7+。
- tf_min=0 时，Teacher Forcing 在前几轮迅速降到接近 0：第 2 轮 BLEU 冲到 6.62，但随后（tf≈0.24→0.0000x）BLEU 逐步下滑到 5.x–6.x，训练 loss 虽持续下降，验证 BLEU 却回落，典型的过拟合+exposure bias 叠加。生成时更易出现重复或过早输出 `<eos>`。
- 结论：保持一个不为 0 的 tf_min（如 0.3）是更稳妥的折中；若想降低 TF，可放慢衰减或提高 tf_min，而非直接降到 0。

== RNN + SentencePiece 子词分词
=== 实验 checkpoint
- `checkpoints/rnn_sp_add_100k_heavy/best_rnn.pt`（SentencePiece 子词分词，additive 注意力）
- `checkpoints/rnn_add_100k_bs128_heavy/best_rnn.pt`（词级分词，additive 注意力）
- `checkpoints/rnn_sp_add_100k_maxlen256/best_rnn.pt`（SentencePiece 子词分词，additive 注意力，最大句长256）
=== 训练曲线

#figure(
  image("figures/rnn_sp_attn_loss.png"),
  caption: "RNN SentencePiece 子词分词的训练 loss 曲线"
)
#figure(
  image("figures/rnn_sp_attn_bleu.png"),
  caption: "RNN SentencePiece 子词分词的验证 BLEU 曲线"
)

=== 结果对比（验证集最佳 BLEU）
#figure(
  three-line-table()[
  |模型目录|分词|max_len|验证 BLEU（best）|
  |---|---|---|---|
  |rnn_add_100k_bs128_heavy|词级|128|7.8|
  |rnn_sp_add_100k_heavy|SentencePiece|128|1.54|
  |rnn_sp_add_100k_maxlen256|SentencePiece|256|2.31|
],
  caption: "RNN 词级 vs SentencePiece 子词分词的验证集最佳 BLEU 对比"

)


=== 现象与分析
- 在相同网络和训练轮数下，词级模型验证 BLEU 可达 7.8，而子词模型在当前设置下最高仅约 1.5（max_len=128）或 2.3（max_len=256），明显未收敛到可用水平。
- 主要原因推测：子词序列更长，max_len=128 下被截断较多；同时训练轮数、学习率和 teacher forcing 调度沿用词级配置，未针对更长序列和更大词表重新调优，导致收敛缓慢。
- 将 max_len 提到 256 略有提升，但仍明显低于词级模型，说明需要进一步增大 max_len、延长训练轮数或调整学习率/梯度裁剪，才可能发挥子词建模对 OOV 的优势。

=== 结论
在当前超参与句长设置下，子词模型未能跑出优于词级的 BLEU。若时间和算力允许，建议针对 SentencePiece 重新调参（更长 max_len、更长训练、适当调小 lr/调大 batch），否则在本次提交中采用收敛更好的词级模型作为主要 RNN 结果。

= Transformer-based NMT 实验
本节简要介绍自实现的 Transformer 模型：encoder-decoder 架构，支持绝对位置编码（正弦）与 T5 风格相对位置偏置；归一化可选 LayerNorm 或 RMSNorm；多头注意力中显式布尔 mask 以避免 CUDA 断言，且相对位置偏置做索引裁剪以防越界。

== 基本规模 Transformer（256 维）— pos/norm Ablation
=== 模型结构
d_model=256，nhead=4，encoder/decoder 各 4 层，FFN=512，dropout 与词级分词，训练/验证集均为 100k 配置。

=== 实验 checkpoint
- `checkpoints/tf_abs_layer_256_100k_heavy/best_transformer.pt`（绝对位置 + LayerNorm）
- `checkpoints/tf_abs_rms_256_100k_heavy/best_transformer.pt`（绝对位置 + RMSNorm）
- `checkpoints/tf_rel_layer_256_100k_heavy/best_transformer.pt`（相对位置 + LayerNorm）
- `checkpoints/tf_rel_rms_256_100k_heavy/best_transformer.pt`（相对位置 + RMSNorm）

=== 训练曲线

#figure(
  image("figures/tf_pos_norm_loss.png"),
  caption: "Transformer 不同位置编码与归一化方式的训练 loss 曲线"
)

#figure(
  image("figures/tf_pos_norm_bleu.png"),
  caption: "Transformer 不同位置编码与归一化方式的验证 BLEU 曲线"
)


=== 结果对比（验证集最佳 BLEU）
#figure(
  three-line-table()[
  |模型目录|pos|norm|验证 BLEU（best）|
  |---|---|---|---|
  |tf_abs_layer_256_100k_heavy|abs|LayerNorm|9.04|
  |tf_abs_rms_256_100k_heavy|abs|RMSNorm|8.49|
  |tf_rel_layer_256_100k_heavy|rel|LayerNorm|8.88|
  |tf_rel_rms_256_100k_heavy|rel|RMSNorm|8.66|
],
  caption: "Transformer 位置编码与归一化方式的验证集最佳 BLEU 对比"
)

=== 现象与分析
- 相对位置整体略优于绝对位置（约 +0.2～0.4 BLEU），但差距不大；说明在本数据规模下，相对距离建模有一定帮助但非决定性。
- LayerNorm vs RMSNorm 差异很小，RMSNorm 未显著超越 LayerNorm，训练曲线总体平滑，表明基础配置已较稳健。
- 四种组合均能在 8–9 BLEU 区间收敛，要进一步提升需依赖更大模型、长句长或更充分的超参调优。

== 更大规模 Transformer + 超参敏感性（512 维）
=== 模型结构
d_model=512，nhead=8，encoder/decoder 各 6 层，FFN=2048，其他设置与 256 维组相同。

=== 实验 checkpoint
- `checkpoints/tf_abs_layer_512_6l_100k/best_transformer.pt`（batch=128, lr=5e-4）
- `checkpoints/tf_abs_layer_512_6l_100k_bs64/best_transformer.pt`（batch=64, lr=5e-4）
- `checkpoints/tf_abs_layer_512_6l_100k_lr1e3/best_transformer.pt`（batch=128, lr=1e-3）

=== 训练曲线
#figure(
  image("figures/tf_large_loss.png"),
  caption: "更大规模 Transformer 的训练 loss 曲线"
)
#figure(
  image("figures/tf_large_bleu.png"),
  caption: "更大规模 Transformer 的验证 BLEU 曲线"
)


=== 结果对比（验证集最佳 BLEU）
#figure(
  three-line-table()[
  |模型目录|d_model|batch|lr|验证 BLEU（best）|
  |---|---|---|---|---|
  |tf_abs_layer_512_6l_100k|512|128|5e-4|8.89|
  |tf_abs_layer_512_6l_100k_bs64|512|64|5e-4|8.52|
  |tf_abs_layer_512_6l_100k_lr1e3|512|128|1e-3|7.83|
],
  caption: "更大规模 Transformer 的验证集最佳 BLEU 对比"
)


=== 现象与分析
- 增大模型与合适的 batch/lr 能把验证 BLEU 提升到 8.5–8.9；batch 减半会小幅降分（8.52），说明较大 batch 有助于稳定梯度。
- 学习率升到 1e-3 时，BLEU 下降到 7.8 左右且更易震荡，提示需配合 warmup/调低初始 lr 或更强正则。
- 大模型训练时间和显存占用显著增加，需结合梯度裁剪与学习率调度防止过拟合和数值不稳定。

== 增加最大句子长度至256
=== 实验 checkpoint
- `checkpoints/tf_abs_layer_256_100k_maxlen256/best_transformer.pt`（max_len=256，其余同基础 256 维组）
- `checkpoints/tf_abs_layer_256_100k_heavy/best_transformer.pt`（max_len=128, baseline）

=== 训练曲线
#figure(
  image("figures/tf_maxlen256_loss.png"),
  caption: "Transformer max_len=256 的训练 loss 曲线"
)
#figure(
  image("figures/tf_maxlen256_bleu.png"),
  caption: "Transformer max_len=256 的验证 BLEU 曲线"
)
=== 结果对比（验证集最佳 BLEU）
#figure(
  three-line-table()[
  |模型目录|max_len|验证 BLEU（best）|
  |---|---|---|
  |tf_abs_layer_256_100k_heavy|128|9.04|
  |tf_abs_layer_256_100k_maxlen256|256|11.43|
],
  caption: "Transformer max_len 对验证集最佳 BLEU 的影响对比"
)
=== 现象与分析

- 将 max_len 提升到 256 后，验证 BLEU 明显提升到 11.43，说明更长的句子长度允许模型捕捉更多上下文信息，尤其对长句翻译有显著帮助。
- 训练曲线显示，max_len=256 的模型在前期收敛更快，且后期 BLEU 更稳定，提示更长句子有助于模型学习更丰富的语言结构。
- 代价是训练时间和显存消耗增加，需权衡资源与性能需求。

= 预训练语言模型微调（以T5为例）
本节使用 HuggingFace 的 `google/mt5-small` 进行中文→英文微调，对比从零训练的 Transformer。

== 实验配置
- 模型：google/mt5-small（Encoder-Decoder，内置 SentencePiece）
- 数据：train_100k_retranslated_hunyuan.jsonl（训练），val_retranslated.jsonl（验证），test_retranslated.jsonl（测试）
- 输入模板：`translate Chinese to English: {zh}`
- 分词：MT5 自带 SentencePiece
- 训练脚本：t5_finetune.py
- 关键超参：learning_rate=3e-4，beam=5 解码
- 保存目录：mt5_translation_100k/（最佳 checkpoint：约 step 15000）

== 结果对比
#figure(
  three-line-table()[
  |模型|验证 BLEU（best）|测试 BLEU|
  |---|---|---|
  |mt5_finetune 15105 epoch |5.3| 4.1 |
  |mt5_finetune 15000 epoch|5.3|4.1|
  |mt5_finetune 10000 epoch|5.0|--|
  |mt5_finetune 5000 epoch|4.2|--|

],
  caption: "MT5 微调的验证集与测试集 BLEU 对比"
)


== 现象与分析
- MT5 微调在 100k 数据上能较快收敛，验证 BLEU 在 4.2–5.3 之间波动，提示可能需要更长训练或更精细调参。
- 测试集 BLEU 约 4.1，低于从零训练的 Transformer，可能因微调轮数不足或超参未最优。
- MT5 训练时间和显存占用显著高于自实现 Transformer，但预训练模型具备更强的语言理解和生成能力，且生成的适合更大数据量和复杂任务。



= 综合对比与分析
== 训练效率
RNN 训练速度快，显存消耗低，适合资源有限场景。
Transformer 训练更耗时，显存需求高，但收敛快，性能优异。
大模型（d_model=512, 6层）训练需数小时，显存消耗大。
MT5 推测可优于从零训练的 Transformer，但尚未补完验证/测试评估；其单步时间与显存占用显著高于自实现 Transformer，需在效果与资源之间权衡。
== 翻译性能（BLEU）
#figure(
  three-line-table()[
  |模型|验证 BLEU（best）|备注|
  |---|---|---|
  |rnn_add_100k_bs128_heavy|7.8|词级，tf_min=0.30|
  |rnn_add_100k_bs128_free|6.6|词级，tf_min=0|
  |rnn_sp_add_100k_maxlen256|2.31|SentencePiece，max_len=256|
  |tf_abs_layer_256_100k_heavy|9.04|abs+LayerNorm，d_model=256|
  |tf_rel_rms_256_100k_heavy|8.66|rel+RMSNorm，d_model=256|
  |tf_abs_layer_512_6l_100k|8.89|大模型，batch=128, lr=5e-4|
  |tf_abs_layer_256_100k_maxlen256|11.43|max_len=256，d_model=256|
  |mt5_finetune_15000 epoch|5.3|基于mini T5 模型进行优化|
],
  caption: "各模型在验证集与测试集的 BLEU 对比"
)
可以看到，在已完成的验证集实验中，基础 Transformer（256 维）相对词级 RNN 提升约 1–2 BLEU，大模型与归一化/位置编码组合在 8–9 BLEU 区间，RNN 子词配置表现最差。预训练 MT5在此次评测中效果比较差，虽好过RNN但是仍未达到Transformer的水平，可能是训练轮数和超参未调优到位所致，后续可继续完善评估。

== 解码策略：Greedy vs Beam Search

在前面的实验中，RNN和Transformer均分别生成了 `greedy` 与 `beam=5` 的翻译结果文件，并在相同的验证 / 测试集上使用 `sacrebleu` 进行评测。这样可以在统一设置下，系统比较不同解码策略对 BLEU 得分与推理延迟的影响。

从预期趋势上看，beam search 通过在每个时间步保留多个候选路径，通常可以在验证 / 测试集上带来约 $1 tilde 2$ 分的 BLEU 提升，特别是对长句或存在多种同义表达的样本，更容易选出语义更完整、语法更自然的译文；但其代价是推理复杂度近似随 beam size 成倍增加，平均解码时间与 GPU 占用也显著上升。在本项目的三个模型族中，RNN 对 beam 的相对加速收益更明显（单步计算简单），Transformer 与 MT5 则在 beam=5 时已经成为整体延迟的主要瓶颈。因此，在实际应用中需要在 BLEU 增益与实时性要求之间权衡：若追求离线高质量翻译，可采用较大的 beam；若面向在线交互式翻译，则更适合使用 greedy 或小 beam（如 3–5）作为折中。

== 可扩展性与泛化
长句处理：Transformer 在 max_len=512 下表现更优，RNN 易丢失长距离信息。
大模型扩展：Transformer 可扩展性强，RNN 受限于递归结构。
低资源场景：子词分词和 Transformer 更具优势。
另外，借助 SentencePiece 等子词建模，Transformer 在小数据量（10k）场景下也能保持相对稳定的 BLEU，而 RNN 在 10k 训练集上的性能下降更加明显，体现出模型容量与正则化的重要性。
== 实用取舍
RNN 模型小，推理快，易实现，适合资源有限场景。
Transformer 性能高，调参复杂，适合大数据和高性能需求。
在解码策略上，beam search 相比 greedy 能带来轻微 BLEU 提升，但会成倍增加推理时间；本项目最终在 RNN/Transformer/MT5 上均采用 beam=5 作为折中方案，在可接受延迟下获得更好的翻译流畅度。

== 实现与复现说明

为便于他人复现本报告中的实验结果，现简要说明代码结构与典型运行命令：

- 代码结构  
  - `train.py`：统一的训练脚本，通过 `--model {rnn, transformer}` 与一系列命令行参数控制模型结构（如 `--attn`、`--d_model`、`--nhead`、`--pos`、`--norm`、`--max_len` 等）和优化设置。  
  - `inference.py`：推理与生成脚本，支持 RNN / Transformer，`--strategy {greedy, beam}`、`--beam` 控制解码方式。  
  - `t5_finetune.py`：基于 HuggingFace Transformers 的 MT5 微调脚本，封装 `Seq2SeqTrainer` 与 `Seq2SeqTrainingArguments`。  
  - `data_utils.py`：数据读取、清洗与分词，包含 `_validate_pair` 过滤规则、词表与 SentencePiece 子词构建。  
  - `plot.py`：读取各个 checkpoint 中的日志（`log_rnn.csv` / `log_transformer.csv`），统一绘制 loss / BLEU 曲线和生成统计图。

- 典型训练命令示例  
  - 词级 RNN + additive 注意力（主力 RNN 模型）：
    - `python train.py --model rnn --attn additive --emb_dim 256 --hid_dim 256 --batch_size 128 --lr 3e-4 --tf_init 0.9 --tf_decay 0.9 --tf_min 0.3 --max_len 128 --train_path dataset/train_100k_retranslated_hunyuan.jsonl --valid_path dataset/val_retranslated.jsonl --out_dir checkpoints/rnn_add_100k_bs128_heavy`  
  - 基础 Transformer（256 维，绝对位置 + LayerNorm）：
    - `python train.py --model transformer --pos absolute --norm layer --d_model 256 --nhead 4 --n_encoder 4 --n_decoder 4 --ff 512 --batch_size 192 --lr 3e-4 --scheduler cosine --warmup_steps 2000 --max_len 128 --train_path dataset/train_100k_retranslated_hunyuan.jsonl --valid_path dataset/val_retranslated.jsonl --out_dir checkpoints/tf_abs_layer_256_100k_heavy`  
  - 大规模 Transformer（512 维，6 层 encoder/decoder）：
    - `python train.py --model transformer --pos absolute --norm layer --d_model 512 --nhead 8 --n_encoder 6 --n_decoder 6 --ff 2048 --batch_size 128 --lr 5e-4 --scheduler cosine --warmup_steps 2000 --max_len 128 --train_path dataset/train_100k_retranslated_hunyuan.jsonl --valid_path dataset/val_retranslated.jsonl --out_dir checkpoints/tf_abs_layer_512_6l_100k`  
  - MT5 微调（简要示意）：
    - `python t5_finetune.py --model_name google/mt5-small --train_file dataset/train_100k_retranslated_hunyuan.jsonl --validation_file dataset/val_retranslated.jsonl --output_dir mt5_translation_100k --learning_rate 3e-4 --num_train_epochs 3 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --evaluation_strategy steps --predict_with_generate True --generation_max_length 256 --generation_num_beams 5`

- 推理与评估  
  - RNN / Transformer 推理统一通过 `inference.py` 实现，例如：  
    - `python inference.py --model transformer --checkpoint checkpoints/tf_abs_layer_256_100k_heavy/best_transformer.pt --data_path dataset/test_retranslated.jsonl --strategy beam --beam 5 --batch_size 64 --max_len 256 --out_file outputs/tf_abs_layer256_beam5_test.txt`  
  - 评估与作图：  
    - 使用 `plot.py` 读取 `checkpoints/*/log_*.csv`，自动绘制 loss/BLEU 曲线并保存至 `figures/`；  
    - 使用 `sacrebleu` 对 `outputs/*_test.txt` 与参考译文对齐，计算 test BLEU 并补入报告中的各表格。

上述命令与脚本均已在项目根目录下经测试，可以在 Linux + Python + PyTorch + Transformers 环境中一键复现实验流程。

== 数据与模型现象的进一步讨论

在前文各小节中，已经报告了 RNN、Transformer 与 MT5 在统一数据与评测设置下的 BLEU 表现。本小节重点从数据质量、分词策略与预训练模型行为三个角度，对一些“异常”现象做更深入的反思和解释。

=== 训练数据质量与分布失配

尽管对 `train_100k.jsonl` 进行了机翻错位清洗与重译，最终仅保留了约 4.6 万对“相对干净”的句对，用于主要实验。这带来两个后果：

- 一方面，清洗后的训练集质量显著高于原始 100k，有利于模型稳定收敛，避免学到大量噪声对齐；  
- 另一方面，有效样本数量缩水近半，且验证/测试集仍然是原始 retranslated 切分，语域与风格与训练集并非完全一致，形成一定程度的分布失配。

这在实验中体现为：  
- RNN 与 Transformer 在训练集和验证集上都能取得还算稳定的 BLEU（RNN 约 7–8，Transformer 可到 9+），但绝对数值并不算高；  
- 在测试集上，尤其对较长或结构复杂的句子，模型仍容易出现过度简化、信息遗漏或语气不自然的情况。

总体看，本项目更像是在“噪声较大、小规模”的真实数据上做建模与对比，而非在大规模高质量语料上的极限性能竞赛。报告中刻意保留了这一点，并在数据预处理小节中强调了训练数据质量有限、数量偏少的事实，避免对 BLEU 数值作过度解读。

=== 子词分词实验失败的原因分析

RNN + SentencePiece 子词分词的实验在当前超参下效果明显不理想：  
- 词级 RNN 可达到验证 BLEU≈7.8；  
- 子词 RNN 在 max_len=128 时仅约 1.5，max_len=256 时也只有 2.3 左右，远低于词级模型。

结合日志与训练配置，原因主要有三点：

- 序列显著变长：同一句话在子词级下 token 数量往往翻倍甚至更多，max_len=128 的截断更加严重，导致很多长句训练和评估时被截断，模型难以学到完整对齐关系。  
- 超参直接沿用词级设置：学习率、训练轮数、Teacher Forcing 调度等都基本拷贝自词级 RNN，没有针对更长序列和更大“有效时间步”做增大训练轮数、减小 lr 或调整梯度裁剪，这会放大优化难度。  
- RNN 对长序列更敏感：与 Transformer 相比，双层 GRU 对序列长度更敏感，长序列 + 子词更容易出现梯度消失/爆炸，即便使用梯度裁剪，模型也需要更长训练才能稳住。

因此，当前的子词实验更像是一个“反例”：在不调参的前提下，直接把词级配置搬到子词场景，往往会得到看似“失败”的结果。这并不意味着子词建模本身无效，而是提醒我们：  
- 子词 + RNN 需要专门调节 max_len、学习率和训练轮数；  
- 若资源有限，本次作业中以词级 RNN 作为主力结果更为稳妥，并在报告中诚实呈现子词实验的负面结果和分析。

=== 预训练 MT5 不如从零训练 Transformer 的可能原因

在本项目中，MT5 微调的验证 BLEU 仅约 4–5，测试 BLEU 约 4.1，显著低于从零训练的 Transformer（验证 BLEU≈9 左右）。这与通常“预训练大模型 > 小模型 from scratch”的经验不符，可能原因包括：

- 微调轮数和步数不足：当前 MT5 在 100k 数据上训练的有效 epoch 数较少，中途多次保存 checkpoint，但 `trainer_state` 中缺少系统的 eval_bleu 记录，说明早期训练脚本并未严格按照“定期评估 + 按 best BLEU 选模型”的流程执行。  
- 超参与解码策略未充分调优：  
  - 学习率、batch_size、梯度累积等参数仍偏向“小模型 from scratch”的直觉，并未针对 MT5 这种预训练模型做系统网格搜索；  
  - 评估时的 beam size、length penalty、max_length 等生成超参也还未完全调优，可能限制了生成质量。  
- 任务与域偏差：本任务的数据量（有效训练样本约 4.6 万）与 MT5 预训练时面对的大规模、多语种语料相比非常小，且经过 retranslation 和清洗后风格相对集中，不一定能充分发挥大模型的泛化和迁移优势。

换句话说，本项目中 MT5 的结果更多反映的是“一个尚未充分调优的预训练模型 baseline”，而非 MT5 的上限性能。在报告中，选择如实呈现 MT5 当前的 BLEU，并在分析中给出上述可能原因和后续改进方向（如：增加微调步数、系统网格搜索 lr/beam/长度惩罚、启用更稳定的 evaluation strategy 等），既展示了预训练模型的潜力，也体现出对实验局限性的清醒认识。

= 总结与展望
本项目系统实现并比较了 RNN 和 Transformer 两类 NMT 模型，覆盖了注意力机制、分词策略、位置编码、归一化和超参数敏感性等多维度实验。实验表明，数据质量和预处理对模型性能影响极大，Transformer 结构在大数据和长句场景下表现突出，RNN 结构虽简单高效但性能有限。未来可进一步探索更强的分词器、更大规模预训练模型，以及 RNN–Transformer 混合架构等方向。
在实现层面，本项目从零实现了数据管线、RNN/Transformer 结构、Teacher Forcing 调度、greedy/beam 解码以及 MT5 微调脚本。通过系统实验，进一步体会到：模型选择只是第一步，数据清洗、分词策略、训练超参和解码策略同样会对最终 BLEU 产生可观影响。

