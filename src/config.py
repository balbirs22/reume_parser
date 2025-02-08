# src/config.py

# Pre-trained BERT model to use for token classification
MODEL_NAME = "bert-base-cased"

# Training hyperparameters
TRAINING_ARGS = {
    "output_dir": "./results",                # Directory where checkpoints and results will be saved
    "evaluation_strategy": "epoch",             # Evaluate the model at the end of each epoch
    "learning_rate": 2e-5,                      # Learning rate for training
    "per_device_train_batch_size": 16,          # Batch size per device during training
    "per_device_eval_batch_size": 16,           # Batch size per device during evaluation
    "num_train_epochs": 3,                      # Number of training epochs
    "weight_decay": 0.01,                       # Weight decay to apply (if any)
}

# Directory where the fine-tuned model and tokenizer will be saved
FINE_TUNED_MODEL_DIR = "./fine_tuned_resume_parser"
