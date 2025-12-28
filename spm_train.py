import argparse
import json
import io
import os
import sentencepiece as spm


def extract_texts(train_jsonl: str, zh_txt: str, en_txt: str):
    assert os.path.exists(train_jsonl), f"No existe: {train_jsonl}"
    with io.open(train_jsonl, "r", encoding="utf-8") as f, \
         io.open(zh_txt, "w", encoding="utf-8") as zf, \
         io.open(en_txt, "w", encoding="utf-8") as ef:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                z = ex.get("zh", "").strip()
                e = ex.get("en", "").strip()
                if z:
                    zf.write(z + "\n")
                if e:
                    ef.write(e + "\n")
            except Exception:
                # saltar líneas corruptas
                continue
    print(f"Wrote {zh_txt} and {en_txt}")


def train_spm(input_txt: str, model_prefix: str, vocab_size: int, coverage: float, model_type: str = "bpe"):
    assert os.path.exists(input_txt), f"No existe: {input_txt}"
    spm.SentencePieceTrainer.Train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=3,
        bos_id=1,
        eos_id=2,
        user_defined_symbols=[]
    )
    print(f"Trained {model_prefix}.model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--zh_txt', type=str, default='spm_zh.txt')
    parser.add_argument('--en_txt', type=str, default='spm_en.txt')
    parser.add_argument('--zh_prefix', type=str, default='spm_zh')
    parser.add_argument('--en_prefix', type=str, default='spm_en')
    parser.add_argument('--zh_vocab', type=int, default=32000)
    parser.add_argument('--en_vocab', type=int, default=32000)
    parser.add_argument('--zh_coverage', type=float, default=0.9995)
    parser.add_argument('--en_coverage', type=float, default=1.0)
    parser.add_argument('--model_type', type=str, choices=['bpe','unigram','char','word'], default='bpe')
    args = parser.parse_args()

    extract_texts(args.train_jsonl, args.zh_txt, args.en_txt)
    train_spm(args.zh_txt, args.zh_prefix, args.zh_vocab, args.zh_coverage, args.model_type)
    train_spm(args.en_txt, args.en_prefix, args.en_vocab, args.en_coverage, args.model_type)


if __name__ == '__main__':
    main()
