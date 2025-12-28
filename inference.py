import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import TranslationDataset, Vocabulary, SubwordVocab, Collate
from rnn_model import EncoderRNN, Attention, DecoderRNN, Seq2Seq
from transformer_model import TransformerNMT


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
        outputs = torch.stack(outputs, dim=1)
        decoded = []
        for b in range(B):
            seq = outputs[b].tolist()
            if eos_idx in seq:
                seq = seq[:seq.index(eos_idx)]
            decoded.append(seq)
        return decoded


def beam_search_rnn(model, src, beam_width, max_len, sos_idx, eos_idx):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        mask = (src != 0).long()
        B = src.size(0)
        results = []
        for b in range(B):
            beams = [(0.0, [sos_idx], hidden[:, b:b+1, :])]
            finished = []
            for _ in range(max_len):
                candidates = []
                for score, seq, h in beams:
                    if seq[-1] == eos_idx:
                        finished.append((score, seq))
                        candidates.append((score, seq, h))
                        continue
                    last = torch.tensor([seq[-1]], device=src.device)
                    # GRU 要求 hidden 为 contiguous
                    logits, new_h = model.decoder(last, h.contiguous(), enc_out[b:b+1], mask[b:b+1])
                    log_probs = F.log_softmax(logits, dim=1)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width, dim=1)
                    for k in range(beam_width):
                        sym = int(topk_ids[0, k].item())
                        new_score = score + float(topk_log_probs[0, k].item())
                        candidates.append((new_score, seq + [sym], new_h))
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
                if all(s[-1] == eos_idx for _, s, _ in beams):
                    break
            best = sorted([(s, seq) for s, seq in finished] + [(sc, se) for sc, se, _ in beams], key=lambda x: x[0], reverse=True)[0][1]
            # 去掉 sos 与截断 eos
            if best and best[0] == sos_idx:
                best = best[1:]
            if eos_idx in best:
                best = best[:best.index(eos_idx)]
            results.append(best)
        return results


def greedy_decode_transformer(model, src, src_mask, max_len, sos_idx, eos_idx):
    model.eval()
    with torch.no_grad():
        # 将 key padding mask 转为布尔类型（True=keep，False=pad）
        src_kpm = (src_mask != 0)
        ys = torch.full((src.size(0), 1), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            tgt_kpm = (ys != 0)
            logits = model(src, src_kpm, ys, tgt_kpm)
            next_tok = logits[:, -1].argmax(dim=1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
        decoded = []
        for b in range(src.size(0)):
            seq = ys[b].tolist()[1:]
            if eos_idx in seq:
                seq = seq[:seq.index(eos_idx)]
            decoded.append(seq)
        return decoded


def beam_search_transformer(model, src, src_mask, beam_width, max_len, sos_idx, eos_idx):
    model.eval()
    with torch.no_grad():
        B = src.size(0)
        # Bool key padding mask
        src_kpm = (src_mask != 0)
        results = []
        for b in range(B):
            beams = [(0.0, [sos_idx])]
            for _ in range(max_len):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == eos_idx:
                        candidates.append((score, seq))
                        continue
                    ys = torch.tensor([seq], device=src.device)
                    tgt_kpm = (ys != 0)
                    logits = model(src[b:b+1], src_kpm[b:b+1], ys, tgt_kpm)
                    log_probs = F.log_softmax(logits[:, -1], dim=1)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width, dim=1)
                    for k in range(beam_width):
                        sym = int(topk_ids[0, k].item())
                        new_score = score + float(topk_log_probs[0, k].item())
                        candidates.append((new_score, seq + [sym]))
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
                if all(s[-1] == eos_idx for _, s in beams):
                    break
            best = beams[0][1]
            if best and best[0] == sos_idx:
                best = best[1:]
            if eos_idx in best:
                best = best[:best.index(eos_idx)]
            results.append(best)
        return results


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get('config', {})

    # 根据 tokenizer 类型构建词表：word 模式复用 checkpoint 内保存的 itos，
    # SentencePiece 模式直接从提供的 spm 模型重建 SubwordVocab，避免旧 checkpoint 的特殊 token 不一致问题。
    if cfg.get('tokenizer', 'word') == 'sp' or getattr(args, 'tokenizer', 'word') == 'sp':
        assert args.sp_src_model and args.sp_tgt_model, "SentencePiece 推理需要 --sp_src_model 与 --sp_tgt_model"
        src_vocab = SubwordVocab(args.sp_src_model, lang='zh')
        tgt_vocab = SubwordVocab(args.sp_tgt_model, lang='en')
    else:
        itos_src = ckpt['src_vocab']
        itos_tgt = ckpt['tgt_vocab']
        src_vocab = Vocabulary(lang='zh')
        tgt_vocab = Vocabulary(lang='en')
        src_vocab.itos = itos_src
        src_vocab.stoi = {tok: idx for idx, tok in src_vocab.itos.items()}
        tgt_vocab.itos = itos_tgt
        tgt_vocab.stoi = {tok: idx for idx, tok in tgt_vocab.itos.items()}

    src_tok = src_vocab.tokenizer
    tgt_tok = tgt_vocab.tokenizer
    ds = TranslationDataset(args.data_path, src_tok, tgt_tok, max_len=args.max_len)
    collate = Collate(src_vocab, tgt_vocab, device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    if args.model == 'rnn':
        encoder = EncoderRNN(len(src_vocab), cfg.get('emb_dim', 256), cfg.get('hid_dim', 256), n_layers=2, dropout=cfg.get('dropout', 0.2))
        attention = Attention(enc_hid_dim=cfg.get('hid_dim', 256), dec_hid_dim=cfg.get('hid_dim', 256), method=cfg.get('attn', 'dot'))
        decoder = DecoderRNN(len(tgt_vocab), cfg.get('emb_dim', 256), cfg.get('hid_dim', 256), cfg.get('hid_dim', 256), n_layers=2, dropout=cfg.get('dropout', 0.2), attention=attention)
        model = Seq2Seq(encoder, decoder, device)
    else:
        model = TransformerNMT(
            src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), d_model=cfg.get('d_model', 256), nhead=cfg.get('nhead', 4),
            num_encoder_layers=cfg.get('n_encoder', 4), num_decoder_layers=cfg.get('n_decoder', 4), dim_feedforward=cfg.get('ff', 512),
            dropout=cfg.get('dropout', 0.1), norm_type=cfg.get('norm', 'layer'), use_relative_bias=(cfg.get('pos', 'absolute')=='relative'), use_abs_pos=(cfg.get('pos', 'absolute')=='absolute')
        )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    sos_idx = tgt_vocab.stoi['<sos>']
    eos_idx = tgt_vocab.stoi['<eos>']

    outputs = []
    with torch.no_grad():
        for src, src_mask, tgt_inp, tgt_out in loader:
            if args.model == 'rnn':
                if args.strategy == 'greedy':
                    hyps = greedy_decode_rnn(model, src, args.max_len, sos_idx, eos_idx)
                else:
                    hyps = beam_search_rnn(model, src, args.beam, args.max_len, sos_idx, eos_idx)
            else:
                if args.strategy == 'greedy':
                    hyps = greedy_decode_transformer(model, src, src_mask, args.max_len, sos_idx, eos_idx)
                else:
                    hyps = beam_search_transformer(model, src, src_mask, args.beam, args.max_len, sos_idx, eos_idx)
            texts = [' '.join([tgt_vocab.itos.get(i, '<unk>') for i in hyp]) for hyp in hyps]
            outputs.extend(texts)

    if args.out_file:
        with open(args.out_file, 'w', encoding='utf-8') as f:
            for line in outputs:
                f.write(line.strip() + '\n')
        print(f"Saved translations to {args.out_file}")
    else:
        for line in outputs[:10]:
            print(line)

def run_inference_pretrained(args):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).cuda()

    # 读取所有中文句子
    src_texts = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            src_texts.append("translate Chinese to English: " + obj['zh'].strip())

    outputs = []
    for i in tqdm(range(0, len(src_texts), args.batch_size)):
        batch = src_texts[i:i+args.batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(model.device)
        gen = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=args.max_len,
            num_beams=args.beam,
            early_stopping=True
        )
        batch_outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend([o.strip() for o in batch_outputs])

    with open(args.out_file, 'w', encoding='utf-8') as f:
        for line in outputs:
            f.write(line + '\n')
    print(f"已保存翻译结果到 {args.out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='dataset/test.jsonl')
    parser.add_argument('--model', type=str, choices=['rnn', 'transformer', 'pretrained'], default='rnn')
    parser.add_argument('--strategy', type=str, choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--out_file', type=str, default='pred.txt')
    parser.add_argument('--tokenizer', type=str, choices=['word', 'sp'], default='word')
    parser.add_argument('--sp_src_model', type=str, default='')
    parser.add_argument('--sp_tgt_model', type=str, default='')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.model == 'pretrained':
        run_inference_pretrained(args)
    else:
        run_inference(args)


if __name__ == '__main__':
    main()