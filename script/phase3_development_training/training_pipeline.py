#!/usr/bin/env python3
"""
Phase 3: Model Development & Training - Training Pipeline Module
Comprehensive model training orchestration with multi-GPU support, optimization, and experiment tracking
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import yaml
from enum import Enum
import warnings

# ML Libraries
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import wandb
import mlflow
import mlflow.pytorch

# Optimization Libraries
import optuna
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import HyperOptSearch

# Local imports
from src.utilities.configuration_manager import ConfigurationManager
from src.utilities.version_control import VersionControlManager

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types"""
    FOUNDATION_MODEL = "foundation_model"
    CUSTOM_ARCHITECTURE = "custom_architecture"
    ENSEMBLE_MODEL = "ensemble_model"
    ADAPTIVE_MODEL = "adaptive_model"


class TrainingStrategy(Enum):
    """Training strategies"""
    STANDARD_FINETUNING = "standard_finetuning"
    LORA_FINETUNING = "lora_finetuning"
    QLORA_FINETUNING = "qlora_finetuning"
    CURRICULUM_LEARNING = "curriculum_learning"
    MULTI_TASK_LEARNING = "multi_task_learning"


@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    # Model Configuration
    model_name_or_path: str
    model_type: ModelType
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = False

    # Training Parameters
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # Memory Optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Distributed Training
    distributed: bool = False
    local_rank: int = -1
    deepspeed_config: Optional[str] = None

    # LoRA Configuration
    use_lora: bool = False
    lora_config: Optional[Dict[str, Any]] = None

    # Evaluation Configuration
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Logging Configuration
    logging_dir: str = "./logs"
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Early Stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3

    # Experiment Tracking
    experiment_name: str = "llm_training"
    mlflow_tracking_uri: str = "http://localhost:5000"
    wandb_project: Optional[str] = None

    # Data Configuration
    train_file: str = ""
    validation_file: str = ""
    test_file: str = ""
    max_length: int = 512
    preprocessing_num_workers: int = 4

    # Custom Parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


class DatasetProcessor:
    """Handles dataset loading and preprocessing"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None

    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer"""
        tokenizer_name = self.config.tokenizer_name or self.config.model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        return tokenizer

    def load_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load training and validation datasets"""
        # Load training data
        if self.config.train_file.endswith('.json') or self.config.train_file.endswith('.jsonl'):
            train_dataset = load_dataset('json', data_files=self.config.train_file, split='train')
        elif self.config.train_file.endswith('.csv'):
            train_dataset = load_dataset('csv', data_files=self.config.train_file, split='train')
        else:
            # Try loading as HuggingFace dataset
            train_dataset = load_dataset(self.config.train_file, split='train')

        # Load validation data if available
        val_dataset = None
        if self.config.validation_file:
            if self.config.validation_file.endswith('.json') or self.config.validation_file.endswith('.jsonl'):
                val_dataset = load_dataset('json', data_files=self.config.validation_file, split='train')
            elif self.config.validation_file.endswith('.csv'):
                val_dataset = load_dataset('csv', data_files=self.config.validation_file, split='train')
            else:
                val_dataset = load_dataset(self.config.validation_file, split='train')

        return train_dataset, val_dataset

    def tokenize_function(self, examples):
        """Tokenize dataset examples"""
        # Handle different text column names
        text_column = None
        for col in ['text', 'content', 'input', 'prompt', 'conversation']:
            if col in examples:
                text_column = col
                break

        if text_column is None:
            raise ValueError("Could not find text column in dataset")

        return self.tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,  # We'll handle padding in the data collator
            max_length=self.config.max_length,
            return_attention_mask=False,
        )

    def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Prepare tokenized datasets"""
        # Load tokenizer
        self.load_tokenizer()

        # Load datasets
        train_dataset, val_dataset = self.load_datasets()

        # Tokenize datasets
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=[col for col in train_dataset.column_names if col not in ['attention_mask', 'input_ids', 'labels']]
        )

        # For language modeling, we set labels to input_ids
        tokenized_train = tokenized_train.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True
        )

        tokenized_val = None
        if val_dataset:
            tokenized_val = val_dataset.map(
                self.tokenize_function,
                batched=True,
                num_proc=self.config.preprocessing_num_workers,
                remove_columns=[col for col in val_dataset.column_names if col not in ['attention_mask', 'input_ids', 'labels']]
            )
            tokenized_val = tokenized_val.map(
                lambda examples: {"labels": examples["input_ids"]},
                batched=True
            )

        return tokenized_train, tokenized_val


class ModelManager:
    """Manages model loading and configuration"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self) -> nn.Module:
        """Load and configure model"""
        # Determine model loading parameters
        load_kwargs = {
            "torch_dtype": torch.float16 if self.config.fp16 else torch.bfloat16 if self.config.bf16 else torch.float32,
            "device_map": "auto" if not self.config.distributed else None,
            "trust_remote_code": self.config.trust_remote_code,
        }

        # Handle quantization for QLoRA
        if self.config.use_lora and self.config.lora_config and self.config.lora_config.get("bits", 0) > 0:
            load_kwargs.update({
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            })

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **load_kwargs
        )

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply LoRA if configured
        if self.config.use_lora:
            self.model = self._apply_lora(self.model)

        return self.model

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA configuration to model"""
        if not self.config.lora_config:
            raise ValueError("LoRA configuration not provided")

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_config.get("r", 16),
            lora_alpha=self.config.lora_config.get("alpha", 32),
            lora_dropout=self.config.lora_config.get("dropout", 0.1),
            target_modules=self.config.lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=self.config.lora_config.get("bias", "none"),
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {}

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "model_type": type(self.model).__name__,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }


class CustomTrainer(Trainer):
    """Custom trainer with additional functionality"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler() if self.args.fp16 else None

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with proper handling of different model types"""
        # Handle PEFT models
        if hasattr(model, 'module'):
            # This is a PEFT model wrapped in DDP
            base_model = model.module
        else:
            base_model = model

        # Forward pass
        with autocast(enabled=self.args.fp16, dtype=torch.float16 if self.args.fp16 else torch.bfloat16 if self.args.bf16 else torch.float32):
            outputs = base_model(**inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Create optimizer with custom configuration"""
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # Separate parameters for LoRA
        if hasattr(self.model, 'peft_config'):
            # PEFT model - optimize only trainable parameters
            trainable_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
            optimizer_kwargs["params"] = trainable_params
        else:
            # Regular model
            optimizer_kwargs["params"] = self.model.parameters()

        return optimizer_cls(**optimizer_kwargs)

    def create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        if self.args.lr_scheduler_type == "cosine":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        else:
            return super().create_scheduler(num_training_steps)


class TrainingPipeline:
    """Main training pipeline orchestrator"""

    def __init__(self, config_path: str):
        self.config_manager = ConfigurationManager(config_path)
        self.config = TrainingConfig(**self.config_manager.get_section("model_development"))

        # Initialize components
        self.dataset_processor = DatasetProcessor(self.config)
        self.model_manager = ModelManager(self.config)

        # Setup output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)

        # Initialize experiment tracking
        self._setup_experiment_tracking()

    def _setup_experiment_tracking(self):
        """Setup MLflow and W&B tracking"""
        # MLflow setup
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

        # W&B setup
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=asdict(self.config)
            )

    def prepare_training_arguments(self) -> TrainingArguments:
        """Prepare HuggingFace TrainingArguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,

            # Memory optimization
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,

            # Evaluation and saving
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,

            # Logging
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,

            # Distributed training
            local_rank=self.config.local_rank,
            deepspeed=self.config.deepspeed_config,

            # Custom params
            **self.config.custom_params
        )

    def run_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("Starting training pipeline")

        # Log experiment start
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log configuration
            mlflow.log_params(asdict(self.config))

            # Prepare datasets
            logger.info("Loading and preparing datasets")
            train_dataset, val_dataset = self.dataset_processor.prepare_datasets()
            logger.info(f"Training dataset size: {len(train_dataset)}")
            if val_dataset:
                logger.info(f"Validation dataset size: {len(val_dataset)}")

            # Load model
            logger.info("Loading model")
            model = self.model_manager.load_model()
            model_info = self.model_manager.get_model_info()
            logger.info(f"Model info: {model_info}")

            # Log model info
            mlflow.log_params(model_info)

            # Prepare data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.dataset_processor.tokenizer,
                mlm=False  # Causal LM
            )

            # Prepare training arguments
            training_args = self.prepare_training_arguments()

            # Setup callbacks
            callbacks = []
            if self.config.early_stopping:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config.early_stopping_patience
                    )
                )

            # Create trainer
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )

            # Train model
            logger.info("Starting training")
            trainer.train()

            # Save final model
            logger.info("Saving final model")
            trainer.save_model()

            # Evaluate model
            eval_results = None
            if val_dataset:
                logger.info("Evaluating model")
                eval_results = trainer.evaluate()
                logger.info(f"Evaluation results: {eval_results}")
                mlflow.log_metrics(eval_results)

            # Prepare results
            results = {
                "model_info": model_info,
                "training_completed": True,
                "eval_results": eval_results,
                "output_dir": self.config.output_dir,
                "run_id": run.info.run_id
            }

            return results

    def run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization using Optuna"""
        logger.info("Starting hyperparameter optimization")

        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            num_epochs = trial.suggest_int('num_epochs', 1, 10)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
            warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.0, 0.3)

            # Update config
            config_dict = asdict(self.config)
            config_dict.update({
                'learning_rate': learning_rate,
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'num_train_epochs': num_epochs,
                'weight_decay': weight_decay,
                'warmup_ratio': warmup_ratio
            })

            # Create temporary config
            temp_config = TrainingConfig(**config_dict)

            # Update pipeline config
            self.config = temp_config

            try:
                # Run training
                results = self.run_training()

                # Get evaluation loss for optimization
                if results['eval_results']:
                    eval_loss = results['eval_results'].get('eval_loss', float('inf'))
                else:
                    eval_loss = float('inf')

                # Log to Optuna
                trial.report(eval_loss, 0)

                return eval_loss

            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf')

        # Create study
        study = optuna.create_study(
            study_name=f"{self.config.experiment_name}_hyperopt",
            direction='minimize'
        )

        # Optimize
        study.optimize(objective, n_trials=10)

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best value: {best_value}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study
        }

    def run_distributed_training(self) -> Dict[str, Any]:
        """Run distributed training setup"""
        logger.info("Setting up distributed training")

        # This would be called with torch.distributed.launch
        # The local_rank would be set by the launcher
        if self.config.local_rank == -1:
            logger.error("Distributed training requires local_rank to be set")
            raise ValueError("local_rank must be set for distributed training")

        return self.run_training()


def setup_distributed():
    """Setup distributed training environment"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size

    return None, None, None


async def main():
    """Main function for CLI execution"""
    parser = argparse.ArgumentParser(description="LLM Training Pipeline")
    parser.add_argument("--config", required=True,
                       help="Configuration file path")
    parser.add_argument("--mode", choices=["train", "hyperopt", "distributed"],
                       default="train", help="Training mode")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")

    args = parser.parse_args()

    # Setup distributed training if needed
    rank, local_rank, world_size = setup_distributed()

    # Initialize pipeline
    pipeline = TrainingPipeline(args.config)

    # Set local rank if distributed
    if args.mode == "distributed" or local_rank is not None:
        pipeline.config.local_rank = local_rank if local_rank is not None else args.local_rank
        pipeline.config.distributed = True

    try:
        if args.mode == "train":
            results = pipeline.run_training()
        elif args.mode == "hyperopt":
            results = pipeline.run_hyperparameter_optimization()
        elif args.mode == "distributed":
            results = pipeline.run_distributed_training()

        logger.info("Training completed successfully")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Clean up distributed training
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(main())