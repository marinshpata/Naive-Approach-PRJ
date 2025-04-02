import os
import random
import math
import numpy as np
import torch

from datasets import Dataset, load_dataset  # https://huggingface.co/docs/datasets
import evaluate  # https://huggingface.co/docs/evaluate
from transformers import (  # https://huggingface.co/docs/transformers
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

MODEL_NAME = "distilbert-base-uncased"
DATASET_DICTIONARY_PATH = "dataset/dictionary.txt"
DATASET_BOOKS_PATH = "dataset/books.txt"
DATASET_SPOKEN_PATH = "dataset/children_spoken_language.train"
TARGET_TOKENS = 3000000 #downsample in case dataset has more than three million
MLM_EPOCHS = 1
MLM_LR = 5e-5
MLM_BATCH_SIZE = 2
GLUE_TASKS = ["mrpc", "sst2", "qnli", "qqp", "rte"]
GLUE_EPOCHS = 3
GLUE_BATCH_SIZE = 16
GLUE_LR = 2e-5
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def create_dataset_from_txt(filepath):
    lines = load_text_file(filepath)
    print(f"Loaded {len(lines)} lines from {filepath}")
    return Dataset.from_dict({"text": lines})

def downsample_by_tokens(dataset, tokenizer, target_token_count):
    text_list = dataset["text"]
    random.shuffle(text_list)
    selected_texts, current_tokens = [], 0

    for txt in text_list:
        token_ids = tokenizer.encode(txt, add_special_tokens=False)
        if current_tokens + len(token_ids) > target_token_count:
            break
        selected_texts.append(txt)
        current_tokens += len(token_ids)

    print(f"Downsampled to {len(selected_texts)} lines (~{current_tokens} tokens).")
    return Dataset.from_dict({"text": selected_texts})

def tokenize_for_mlm(example, tokenizer, max_len=128):  # https://huggingface.co/docs/transformers/tasks/masked_language_modeling
    enc = tokenizer(example["text"], truncation=True, max_length=max_len, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

def domain_adapt_no_eval(dataset, tokenizer, base_model_ckpt, output_dir, epochs, batch_size, lr, seed=42):
    model = (AutoModelForMaskedLM.from_pretrained(base_model_ckpt)
             if isinstance(base_model_ckpt, str) else base_model_ckpt)
    ds_tokenized = dataset.map(lambda ex: tokenize_for_mlm(ex, tokenizer),
                               batched=True, remove_columns=["text"])  # https://huggingface.co/docs/datasets/v2.0.0/en/package_reference/main_classes#datasets.Dataset.map

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15) # https://huggingface.co/docs/transformers/tasks/masked_language_modeling

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=lr,
        load_best_model_at_end=False,
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(  # https://huggingface.co/docs/transformers/main_classes/trainer
        model=model,
        args=training_args,
        train_dataset=ds_tokenized,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Domain-adapted model saved to: {output_dir}")

#https://huggingface.co/docs/transformers/tasks/sequence_classification - for all pre-proccess functions
def preprocess_glue_sst2(example, tokenizer):
    return tokenizer(example["sentence"], truncation=True, max_length=128)

def preprocess_glue_qnli(example, tokenizer):
    return tokenizer(example["question"], example["sentence"], truncation=True, max_length=128)

def preprocess_glue_qqp(example, tokenizer):
    return tokenizer(example["question1"], example["question2"], truncation=True, max_length=128)

def preprocess_glue_rte(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, max_length=128)

def preprocess_glue_generic(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, max_length=128)

def get_preprocessor(task):
    if task == "sst2":
        return preprocess_glue_sst2
    elif task == "qnli":
        return preprocess_glue_qnli
    elif task == "qqp":
        return preprocess_glue_qqp
    elif task == "rte":
        return preprocess_glue_rte
    else:
        return preprocess_glue_generic

def glue_finetune_and_eval(domain_adapted_ckpt, glue_task, epochs, batch_size, lr, seed=42): #https://huggingface.co/learn/nlp-course/en/chapter3/3
    raw_glue = load_dataset("glue", glue_task)
    train_ds = raw_glue["train"]
    valid_ds = raw_glue["validation"]

    tokenizer = AutoTokenizer.from_pretrained(domain_adapted_ckpt, use_fast=True)

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(domain_adapted_ckpt, num_labels=num_labels)

    preprocessor = get_preprocessor(glue_task)
    encoded_train = train_ds.map(lambda x: preprocessor(x, tokenizer), batched=True)
    encoded_val = valid_ds.map(lambda x: preprocessor(x, tokenizer), batched=True)

    to_remove = list(set(train_ds.column_names) - {"label"})
    encoded_train = encoded_train.remove_columns(to_remove).rename_column("label", "labels")
    encoded_val = encoded_val.remove_columns(to_remove).rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric_glue = evaluate.load("glue", glue_task)

    def compute_glue_metrics(eval_pred): #https://huggingface.co/docs/transformers/tasks/token_classification
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric_glue.compute(predictions=preds, references=labels)

    out_dir = f"./glue_{glue_task}_{os.path.basename(domain_adapted_ckpt)}"
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_glue_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"[GLUE {glue_task}] Results from {domain_adapted_ckpt}: {results}\n")
    return results

def baseline_finetune_on_glue(model_name, glue_task, epochs, batch_size, lr, seed=42):
    raw_glue = load_dataset("glue", glue_task)
    train_ds = raw_glue["train"]
    valid_ds = raw_glue["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    preprocessor = get_preprocessor(glue_task)

    encoded_train = train_ds.map(lambda x: preprocessor(x, tokenizer), batched=True)
    encoded_val = valid_ds.map(lambda x: preprocessor(x, tokenizer), batched=True)

    to_remove = list(set(train_ds.column_names) - {"label"})
    encoded_train = encoded_train.remove_columns(to_remove).rename_column("label", "labels")
    encoded_val = encoded_val.remove_columns(to_remove).rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("glue", glue_task)

    def compute_metrics(eval_pred): #https://huggingface.co/docs/transformers/tasks/token_classification
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    out_dir = f"./glue_baseline_{glue_task}"
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"[BASELINE {glue_task}] -> {model_name}: {results}\n")
    return results

if __name__ == "__main__":
    set_seed(SEED)

    # Dictionaries to store baseline and domain-adapted results
    baseline_results = {}
    domain_results = {
        "dictionary": {},
        "books": {},
        "spoken": {},
    }

    print("===== (A) BASELINE DistilBERT -> GLUE =====\n")
    #baseline results
    for task in GLUE_TASKS:
        res = baseline_finetune_on_glue(
            model_name=MODEL_NAME,
            glue_task=task,
            epochs=GLUE_EPOCHS,
            batch_size=GLUE_BATCH_SIZE,
            lr=GLUE_LR,
            seed=SEED
        )
        baseline_results[task] = res

    print("\n===== DOMAIN ADAPT (Dictionary) =====")
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) #https://huggingface.co/docs/transformers/tasks/token_classification
    dict_output = "./domain_adapt_mlm/dictionary"
    os.makedirs(dict_output, exist_ok=True)
    dict_ds = create_dataset_from_txt(DATASET_DICTIONARY_PATH)
    dict_ds = downsample_by_tokens(dict_ds, base_tokenizer, TARGET_TOKENS)
    domain_adapt_no_eval(dict_ds, base_tokenizer, MODEL_NAME, dict_output, MLM_EPOCHS, MLM_BATCH_SIZE, MLM_LR, SEED)

    print("\n===== DOMAIN ADAPT (Books) =====")
    books_output = "./domain_adapt_mlm/books"
    os.makedirs(books_output, exist_ok=True)
    books_ds = create_dataset_from_txt(DATASET_BOOKS_PATH)
    books_ds = downsample_by_tokens(books_ds, base_tokenizer, TARGET_TOKENS)
    domain_adapt_no_eval(books_ds, base_tokenizer, MODEL_NAME, books_output, MLM_EPOCHS, MLM_BATCH_SIZE, MLM_LR, SEED)

    print("\n===== DOMAIN ADAPT (Children's Spoken) =====")
    spoken_output = "./domain_adapt_mlm/spoken"
    os.makedirs(spoken_output, exist_ok=True)
    spoken_ds = create_dataset_from_txt(DATASET_SPOKEN_PATH)
    spoken_ds = downsample_by_tokens(spoken_ds, base_tokenizer, TARGET_TOKENS)
    domain_adapt_no_eval(spoken_ds, base_tokenizer, MODEL_NAME, spoken_output, MLM_EPOCHS, MLM_BATCH_SIZE, MLM_LR, SEED)

    print("\n===== (C) GLUE EVAL for Each Domain-Adapted Model =====\n")
    domain_ckpts = {
        "dictionary": dict_output,
        "books": books_output,
        "spoken": spoken_output,
    }

    # Domain-Adapted results
    for domain_name, ckpt in domain_ckpts.items():
        print(f"\n>>> FINE-TUNING ON GLUE using {domain_name.upper()}-adapted DistilBERT ({ckpt})\n")
        for task in GLUE_TASKS:
            res = glue_finetune_and_eval(
                domain_adapted_ckpt=ckpt,
                glue_task=task,
                epochs=GLUE_EPOCHS,
                batch_size=GLUE_BATCH_SIZE,
                lr=GLUE_LR,
                seed=SEED
            )
            domain_results[domain_name][task] = res

    print("\n\n===== FINAL CONSOLIDATED RESULTS =====\n")

    print("== Baseline Results (DistilBERT) ==")
    for task, res in baseline_results.items():
        print(f"  Task: {task} => {res}")

    print("\n== Domain-Adapted Results ==")
    for domain_name, tasks_dict in domain_results.items():
        print(f"\n[Domain: {domain_name}]")
        for task, res in tasks_dict.items():
            print(f"  Task: {task} => {res}")

#Hugging face tutorials in natural language processing & transformers
# have been particularly useful in this implementation.