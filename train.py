import argparse
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sacrebleu

from data_utils import TranslationDataset, Vocabulary, SubwordVocab, Collate
from rnn_model import EncoderRNN, Attention, DecoderRNN, Seq2Seq
from transformer_model import TransformerNMT


def build_vocabs(train_jsonl, freq_threshold=2):
    # 读取文本以构建词表
    zh_texts, en_texts = [], []
    with open(train_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = __import__('json').loads(line)
                zh_texts.append(ex['zh'].strip())
                en_texts.append(ex['en'].strip())
            except Exception:
                continue

    src_vocab = Vocabulary(freq_threshold=freq_threshold, lang='zh')
    tgt_vocab = Vocabulary(freq_threshold=freq_threshold, lang='en')
    src_vocab.build_vocabulary(zh_texts)
    tgt_vocab.build_vocabulary(en_texts)
    return src_vocab, tgt_vocab


def greedy_decode_rnn(model, src, max_len, sos_idx, eos_idx):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        mask = (src != 0).long()
        B = src.size(0)
        input_tok = torch.full((B,), sos_idx, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            logits, hidden = model.decoder(input_tok, hidden, enc_out, mask)
            input_tok = logits.argmax(dim=1)
            outputs.append(input_tok)
        outputs = torch.stack(outputs, dim=1)  # [B, L]
        # 截断到 EOS
        decoded = []
        for b in range(B):
            seq = outputs[b].tolist()
            if eos_idx in seq:
                seq = seq[:seq.index(eos_idx)]
            decoded.append(seq)
        return decoded


def greedy_decode_transformer(model, src, src_mask, max_len, sos_idx, eos_idx):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_key_padding_mask=src_mask)
        B = src.size(0)
        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            # 使用布尔型 key padding mask (True=keep)，避免 masked_fill 报 dtype 错误
            logits = model(src, src_mask, ys, (ys != 0))
            next_token = logits[:, -1].argmax(dim=1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        decoded = []
        for b in range(B):
            seq = ys[b].tolist()[1:]  # remove sos
            if eos_idx in seq:
                seq = seq[:seq.index(eos_idx)]
            decoded.append(seq)
        return decoded


def ids_to_text(ids_batch, vocab):
    return [' '.join([vocab.itos.get(i, '<unk>') for i in ids]) for ids in ids_batch]


def evaluate_bleu(model_type, model, loader, tgt_vocab, max_len):
    # refs: lista de referencias (strings); hyps: lista de hipótesis (strings)
    refs, hyps = [], []
    model_core = model.module if hasattr(model, 'module') else model
    sos_idx = tgt_vocab.stoi['<sos>']
    eos_idx = tgt_vocab.stoi['<eos>']
    for src, src_mask, tgt_inp, tgt_out in loader:
        if model_type == 'rnn':
            batch_ids = greedy_decode_rnn(model_core, src, max_len, sos_idx, eos_idx)
        else:
            # src_mask aquí es 1/0; convertir a bool keep-mask para forward
            src_kpm = (src_mask != 0)
            batch_ids = greedy_decode_transformer(model_core, src, src_kpm, max_len, sos_idx, eos_idx)
        hyps.extend(ids_to_text(batch_ids, tgt_vocab))
        # 参考：从 tgt_out 恢复文本
        ref_ids = []
        for b in range(tgt_out.size(0)):
            seq = tgt_out[b].tolist()
            if eos_idx in seq:
                seq = seq[:seq.index(eos_idx)]
            ref_ids.append(seq)
        refs.extend(ids_to_text(ref_ids, tgt_vocab))
    # SacreBLEU espera [ref_streams]; para una sola referencia usamos [refs]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return bleu


def _build_vocabs_from_args(args):
    if args.tokenizer == 'sp':
        assert args.sp_src_model and args.sp_tgt_model, "使用 SentencePiece 时需提供 --sp_src_model 与 --sp_tgt_model"
        src_vocab = SubwordVocab(args.sp_src_model, lang='zh')
        tgt_vocab = SubwordVocab(args.sp_tgt_model, lang='en')
        return src_vocab, tgt_vocab
    else:
        return build_vocabs(args.train_path, freq_threshold=args.freq_threshold)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # 构建词表
    src_vocab, tgt_vocab = _build_vocabs_from_args(args)

    # 数据集与加载器
    src_tok = src_vocab.tokenizer
    tgt_tok = tgt_vocab.tokenizer
    train_ds = TranslationDataset(args.train_path, src_tok, tgt_tok, max_len=args.max_len)
    valid_ds = TranslationDataset(args.valid_path, src_tok, tgt_tok, max_len=args.max_len)
    collate = Collate(src_vocab, tgt_vocab, device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 模型
    if args.model == 'rnn':
        encoder = EncoderRNN(len(src_vocab), args.emb_dim, args.hid_dim, n_layers=2, dropout=args.dropout)
        attention = Attention(enc_hid_dim=args.hid_dim, dec_hid_dim=args.hid_dim, method=args.attn)
        decoder = DecoderRNN(len(tgt_vocab), args.emb_dim, args.hid_dim, args.hid_dim, n_layers=2, dropout=args.dropout, attention=attention)
        model = Seq2Seq(encoder, decoder, device).to(device)
    else:
        model = TransformerNMT(
            src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), d_model=args.d_model, nhead=args.nhead,
            num_encoder_layers=args.n_encoder, num_decoder_layers=args.n_decoder, dim_feedforward=args.ff, dropout=args.dropout,
            norm_type=args.norm, use_relative_bias=(args.pos=='relative'), use_abs_pos=(args.pos=='absolute')
        ).to(device)

    if args.data_parallel and torch.cuda.device_count() > 1 and not args.cpu:
        model = nn.DataParallel(model)
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")

    model_core = model.module if hasattr(model, 'module') else model

    # 训练配置
    pad_idx = tgt_vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 预设 scheduler（warmup + linear/cosine）
    global_step = 0
    scheduler = None
    total_steps = None

    teacher_forcing = args.tf_init
    best_bleu = 0.0
    # Preparar logging CSV
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"log_{args.model}.csv")
    write_header = not os.path.exists(log_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        if scheduler is None and args.scheduler != 'none':
            total_steps = args.epochs * max(1, len(train_loader))
            warmup = max(0, args.warmup_steps)
            def lr_lambda(step):
                if step < warmup:
                    return max(1e-6, step / float(max(1, warmup)))
                progress = (step - warmup) / float(max(1, total_steps - warmup))
                if args.scheduler == 'linear':
                    return max(0.0, 1.0 - progress)
                elif args.scheduler == 'cosine':
                    import math
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                else:
                    return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for src, src_mask, tgt_inp, tgt_out in train_loader:
            optimizer.zero_grad()

            # 保证 masks 为 bool (True=keep)
            src_kpm = (src_mask != 0)
            tgt_kpm = (tgt_inp != pad_idx)

            if args.debug_assert:
                assert src.min() >= 0 and src.max() < len(src_vocab), "src token out of range"
                assert tgt_inp.min() >= 0 and tgt_inp.max() < len(tgt_vocab), "tgt token out of range"

            if args.model == 'rnn':
                # 将 <sos> + y 作为输入，模型在步骤 t=1..T-1 产生预测
                logits = model(src, tgt_inp, teacher_forcing_ratio=teacher_forcing)
                # 对齐目标仅取前 T-1 步以匹配 logits[:, 1:]
                target = tgt_out[:, :logits.size(1) - 1]
                loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), target.reshape(-1))
            else:
                logits = model(src, src_kpm, tgt_inp, tgt_kpm)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            total_loss += loss.item()

        # Teacher Forcing 概率衰减
        teacher_forcing = max(args.tf_min, teacher_forcing * args.tf_decay)

        # 验证 BLEU
        bleu = evaluate_bleu(args.model, model, valid_loader, tgt_vocab, args.max_len)

        elapsed = time.time() - start
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, BLEU={bleu:.2f}, TF={teacher_forcing:.3f}, time={elapsed:.1f}s")
        # Escribir al CSV
        try:
            import csv
            with open(log_path, 'a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['epoch', 'loss', 'bleu', 'tf', 'time'])
                    write_header = False
                w.writerow([epoch, avg_loss, bleu, teacher_forcing, round(elapsed, 3)])
        except Exception as e:
            print(f"[WARN] No se pudo escribir log CSV: {e}")

        # 保存最好模型
        if bleu > best_bleu:
            best_bleu = bleu
            save_path = os.path.join(args.out_dir, f"best_{args.model}.pt")
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                'model': model_core.state_dict(),
                'src_vocab': src_vocab.itos,
                'tgt_vocab': tgt_vocab.itos,
                'config': vars(args)
            }, save_path)
            print(f"Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='dataset/train_10k.jsonl')
    parser.add_argument('--valid_path', type=str, default='dataset/valid.jsonl')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--model', type=str, choices=['rnn', 'transformer'], default='rnn')
    parser.add_argument('--attn', type=str, choices=['dot', 'general', 'concat', 'additive'], default='dot')
    parser.add_argument('--norm', type=str, choices=['layer', 'rms'], default='layer')
    parser.add_argument('--pos', type=str, choices=['absolute', 'relative'], default='absolute')

    parser.add_argument('--freq_threshold', type=int, default=2)
    parser.add_argument('--tokenizer', type=str, choices=['word', 'sp'], default='word')
    parser.add_argument('--sp_src_model', type=str, default='')
    parser.add_argument('--sp_tgt_model', type=str, default='')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scheduler', type=str, choices=['none', 'linear', 'cosine'], default='none')
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--cpu', action='store_true')

    # RNN
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--hid_dim', type=int, default=256)

    # Transformer
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--n_encoder', type=int, default=4)
    parser.add_argument('--n_decoder', type=int, default=4)
    parser.add_argument('--ff', type=int, default=512)

    # Teacher Forcing schedule
    parser.add_argument('--tf_init', type=float, default=1.0)
    parser.add_argument('--tf_decay', type=float, default=0.7)
    parser.add_argument('--tf_min', type=float, default=0.0)

    parser.add_argument('--data_parallel', action='store_true', help='Use nn.DataParallel when multiple GPUs visible')
    parser.add_argument('--debug_assert', action='store_true', help='Enable range asserts for tokens')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()