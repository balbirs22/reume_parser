# src/skill_extractor.py

import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from src.config import MODEL_NAME, TRAINING_ARGS, FINE_TUNED_MODEL_DIR

def tokenize_and_align_labels(examples):
    """
    Tokenizes texts and aligns NER labels with tokenized outputs.
    Expects each example to have "tokens" and "ner_tags".
    """
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_model():
    # Load a sample dataset (using CoNLL2003 as a placeholder)
    dataset = load_dataset("conll2003")
    
    # Tokenize and align labels
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    
    label_list = dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_list)
    
    # Load the pre-trained model for token classification
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    
    training_args = TrainingArguments(
        output_dir=TRAINING_ARGS["output_dir"],
        evaluation_strategy=TRAINING_ARGS["evaluation_strategy"],
        learning_rate=TRAINING_ARGS["learning_rate"],
        per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
        num_train_epochs=TRAINING_ARGS["num_train_epochs"],
        weight_decay=TRAINING_ARGS["weight_decay"],
    )
    
    metric = load_metric("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
    print("Model fine-tuning complete. Saved to", FINE_TUNED_MODEL_DIR)

if __name__ == "__main__":
    train_model()
