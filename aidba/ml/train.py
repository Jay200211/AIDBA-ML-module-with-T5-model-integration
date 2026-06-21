"""Training script for T5 SQL generation model.

Compatible with transformers >= 4.36 (uses processing_class instead of tokenizer).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("aidba.ml.train")


def train_model(
    train_data_path: str,
    eval_data_path: str = None,
    output_dir: str = "./models/aidba-sql-t5",
    base_model: str = "t5-small",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_input_length: int = 512,
    max_output_length: int = 256,
    save_steps: int = 500
):
    """Train the T5 model on SQL data."""
    try:
        import torch
        from transformers import (
            T5ForConditionalGeneration, T5Tokenizer,
            Trainer, TrainingArguments,
            DataCollatorForSeq2Seq
        )
        from torch.utils.data import Dataset
        from aidba.ml.dataset import SpiderDataset, SQLExample
    except ImportError as e:
        log.error(f"Missing dependencies: {e}")
        log.error("Install with: pip install torch transformers datasets")
        return False

    # Detect transformers version to choose correct API
    import transformers
    tf_version = transformers.__version__
    log.info(f"Using transformers version: {tf_version}")
    use_processing_class = tuple(map(int, tf_version.split('.')[:2])) >= (4, 36)

    log.info(f"Loading model: {base_model}")
    try:
        tokenizer = T5Tokenizer.from_pretrained(base_model)
        model = T5ForConditionalGeneration.from_pretrained(base_model)
    except Exception as e:
        log.error(f"Failed to load base model '{base_model}': {e}")
        log.error("Make sure you have internet connection to download the model")
        return False

    # Load dataset
    log.info(f"Loading training data from: {train_data_path}")
    dataset_loader = SpiderDataset()

    if train_data_path.endswith(".jsonl"):
        train_examples = dataset_loader.load_jsonl(train_data_path)
    elif train_data_path.endswith(".json"):
        train_examples = dataset_loader.load_from_spider_json(train_data_path)
    else:
        log.error("Unsupported file format. Use .jsonl or .json")
        return False

    if not train_examples:
        log.error("No training examples loaded!")
        return False

    log.info(f"Loaded {len(train_examples)} training examples")

    # Load eval data if provided
    eval_examples = []
    if eval_data_path:
        if eval_data_path.endswith(".jsonl"):
            eval_examples = dataset_loader.load_jsonl(eval_data_path)
        else:
            eval_examples = dataset_loader.load_from_spider_json(eval_data_path)
        log.info(f"Loaded {len(eval_examples)} eval examples")

    class SQLDataset(Dataset):
        def __init__(self, examples, tokenizer, max_in, max_out):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_in = max_in
            self.max_out = max_out

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]

            # Input: question + schema
            if ex.schema:
                input_text = f"translate English to SQL: {ex.question} | schema: {ex.schema}"
            else:
                input_text = f"translate English to SQL: {ex.question}"

            # Output: SQL query
            output_text = ex.sql

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_in,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            targets = self.tokenizer(
                output_text,
                max_length=self.max_out,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            labels = targets["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }

    train_dataset = SQLDataset(train_examples, tokenizer, max_input_length, max_output_length)
    eval_dataset = SQLDataset(eval_examples, tokenizer, max_input_length, max_output_length) if eval_examples else None

    # Training arguments - using compatible parameters
    eval_strategy_value = "steps" if eval_dataset else "no"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=min(100, len(train_examples) // batch_size),
        weight_decay=0.01,
        logging_steps=5,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy=eval_strategy_value,
        eval_steps=save_steps if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        report_to="none",  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),
        # Avoid deprecated logging_dir warning
        logging_dir=None,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_input_length
    )

    # Trainer - use correct parameter name based on transformers version
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    # Newer transformers uses 'processing_class', older uses 'tokenizer'
    if use_processing_class:
        trainer_kwargs["processing_class"] = tokenizer
        log.info("Using 'processing_class' parameter (transformers >= 4.36)")
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        log.info("Using 'tokenizer' parameter (transformers < 4.36)")

    trainer = Trainer(**trainer_kwargs)

    # Train
    log.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        try:
            trainer.save_model(f"{output_dir}/interrupted")
        except Exception:
            pass
        return True
    except Exception as e:
        log.exception(f"Training failed: {e}")
        return False

    # Save final model
    log.info(f"Saving final model to: {output_dir}")
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        log.warning(f"Failed to save model: {e}")

    log.info("✅ Training complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 SQL model")
    parser.add_argument("--train-data", required=True, help="Path to training data (.jsonl or .json)")
    parser.add_argument("--eval-data", help="Path to eval data")
    parser.add_argument("--output-dir", default="./models/aidba-sql-t5", help="Output directory")
    parser.add_argument("--base-model", default="t5-small", help="Base model (t5-small, t5-base, t5-large)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    success = train_model(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    sys.exit(0 if success else 1)
