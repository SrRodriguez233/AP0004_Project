import argparse
import numpy as np
import sacrebleu
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    # 生成预测文本
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # -100 → pad
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    return {"bleu": bleu.score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/mt5-small')
    parser.add_argument('--train_path', type=str, default='dataset/train_10k.jsonl')
    parser.add_argument('--valid_path', type=str, default='dataset/valid.jsonl')
    parser.add_argument('--output_dir', type=str, default='./mt5_translation')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 注意：train 与 valid 的 JSONL 列名略有不同（例如 zh_hy vs zh_nllb），
    # 直接在一个 load_dataset 调用里会因为 schema 不一致而报错。
    # 因此分别加载，再组合成 DatasetDict。
    train_ds = load_dataset('json', data_files={'train': args.train_path})['train']
    valid_ds = load_dataset('json', data_files={'validation': args.valid_path})['validation']
    dataset = DatasetDict({'train': train_ds, 'validation': valid_ds})

    def preprocess_function(examples):
        # 只依赖通用字段 zh / en，其它多余列（zh_hy, zh_nllb, index）会在 map 时被移除
        inputs = ["translate Chinese to English: " + z for z in examples['zh']]
        targets = examples['en']
        model_inputs = tokenizer(inputs, max_length=args.max_len, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_len, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # 分别对 train / validation 调用 map，并各自移除本 split 的原始列，避免列名不一致导致报错
    tokenized_train = dataset['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
    )
    tokenized_valid = dataset['validation'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['validation'].column_names,
    )
    tokenized = DatasetDict({'train': tokenized_train, 'validation': tokenized_valid})

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # 为兼容当前 transformers 版本，仅使用核心参数，去掉 evaluation_strategy/predict_with_generate 等新字段
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    trainer.train()


if __name__ == '__main__':
    main()