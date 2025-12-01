"""
차트 질문 답변을 위한 Vision Language Model 파인튜닝 스크립트 (Native Fine-tuning 버전)
파일명: train.py
"""

import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
# PEFT 관련 import 제거 (Native Fine-tuning이므로)
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM, # Qwen3VL 또는 Qwen2.5VL 로드용
    Qwen2_5_VLForConditionalGeneration # 사용 가능한 경우
)
from trl import SFTConfig
from toonVLM.weight_utils import WeightedVLDataCollator, WeightedLossTrainer
import wandb

# 모델 클래스 동적 할당
try:
    ModelClass = Qwen2_5_VLForConditionalGeneration
except NameError:
    ModelClass = AutoModelForCausalLM

@dataclass
class ModelArguments:
    model_id: str = field(default="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated")
    processor_id: str = field(default="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated")
    min_pixels: int = field(default=256 * 28 * 28)
    max_pixels: int = field(default=960 * 28 * 28)

@dataclass
class DataArguments:
    """데이터셋 경로 및 손실 가중치 설정"""
    data_path: str = field(default='/workspace/Toonspace_VLM/data/ocr_description/total(bbox_normal)_ocr_dataset_2F.json')
    train_ratio: float = 0.95
    eval_ratio: float = 0.025
    
    # CLI에서 변경할 수 있도록 field로 정의
    text_weight: float = field(
        default=2.0, 
        metadata={"help": "Weight for text generation loss"}
    )
    bbox_weight: float = field(
        default=0.1, 
        metadata={"help": "Weight for bounding box regression loss"}
    )
@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(default="ex_models/output")
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-6)
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=50)
    
    early_stopping_patience: int = field(default=10)
    wandb_project: str = field(default="qwen3-Webtoon-vlm")
    wandb_name: str = field(default="run")

    # [수정] 아래 3줄을 추가하여 EarlyStopping 기준을 정의합니다.
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True)

    eval_strategy: str = field(default="steps")   # 필수: "no" -> "steps"
    save_strategy: str = field(default="steps")   # 필수: eval_strategy와 일치해야 함
    save_steps: int = field(default=100000)
    eval_steps: int = field(default=500)


    # --- Custom Arguments ---
    stage: str = field(
        default="step1", 
        metadata={"help": "Training stage: step1(Internal Adapter), step2(Decoder), step3(Full)"}
    )
    resume_from_prev_stage: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the FULL model checkpoint from the previous stage"}
    )

class VLMTrainer:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.setup_logging()
        self.model = None
        self.processor = None
        
    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        logging.getLogger("codecarbon").setLevel(logging.WARNING)

    def load_model_and_processor(self) -> None:
        self.logger.info(f"Loading Processor: {self.model_args.processor_id}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_args.processor_id,
            min_pixels=self.model_args.min_pixels,
            max_pixels=self.model_args.max_pixels
        )

        # 모델 로드 경로 결정 (이전 단계 결과가 있으면 거기서 로드)
        load_path = self.model_args.model_id
        if self.training_args.resume_from_prev_stage:
            load_path = self.training_args.resume_from_prev_stage
            self.logger.info(f"Resuming from previous stage checkpoint: {load_path}")
        else:
            self.logger.info(f"Loading Base Model: {self.model_args.model_id}")

        self.model = ModelClass.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # 단계별 Freezing 설정 적용
        self.configure_stage()

    def configure_stage(self):
        stage = self.training_args.stage
        self.logger.info(f"Configuring model for Stage: {stage}")

        # 모든 파라미터 일단 Freeze (나중에 필요한 것만 켬)
        # Step 3는 예외
        if stage != "step3":
            self.model.requires_grad_(False)

        if stage == "step1":
            # --- Step 1: Internal Adapter (Merger) Only ---
            self.logger.info("Stage 1: Training Internal Adapter (Merger) ONLY.")
            
            # Qwen2-VL 구조: model.visual.merger (또는 attn_pool, projection 등)
            # 이름에 'merger', 'projector', 'adapter' 등이 들어간 레이어만 Unfreeze
            target_keywords = ['merger', 'projector', 'adapter', 'mlp'] 
            
            trainable_count = 0
            for name, param in self.model.named_parameters():
                # Visual 모듈 내부에 있으면서 연결부 역할을 하는 놈 찾기
                # 보통 visual.merger 형태임.
                if 'visual' in name and any(k in name for k in target_keywords):
                    param.requires_grad = True
                    trainable_count += 1
            
            if trainable_count == 0:
                self.logger.warning("Warning: No adapter/merger layers found with keywords. Checking specific Qwen-VL structure...")
                # Qwen-VL 구체적 대응 (visual.merger)
                if hasattr(self.model, "visual") and hasattr(self.model.visual, "merger"):
                    for param in self.model.visual.merger.parameters():
                        param.requires_grad = True
                    self.logger.info("Unfrozen visual.merger manually.")

        elif stage == "step2":
            # --- Step 2: Adapter + Decoder (Vision Backbone Frozen) ---
            self.logger.info("Stage 2: Training Adapter + Decoder (LLM). Vision Backbone Frozen.")
            
            # 1. LLM (Decoder) 전체 Unfreeze
            # 보통 self.model.model (LLM part) or self.model.layers
            if hasattr(self.model, "model"): # Qwen2 기반
                for name, param in self.model.model.named_parameters():
                    # visual 부분 제외하고 LLM 부분만 (구조에 따라 model 안에 visual이 없을 수도 있고 있을 수도 있음)
                    if "visual" not in name:
                        param.requires_grad = True
            
            # 2. Vision 쪽 Adapter(Merger) Unfreeze
            target_keywords = ['merger', 'projector', 'adapter']
            for name, param in self.model.named_parameters():
                if 'visual' in name and any(k in name for k in target_keywords):
                    param.requires_grad = True

        elif stage == "step3":
            # --- Step 3: Full Fine-tuning ---
            self.logger.info("Stage 3: Unfreezing EVERYTHING (Full Fine-tuning).")
            self.model.requires_grad_(True)

        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"\n\n {'+'*30}Trainable params: {trainable_params:,} / {all_params:,} ({trainable_params/all_params:.2%}){'+'*30}\n\n")

    def format_data(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        system_msg = """당신은 웹툰 이미지 분석 전문가입니다. 성인 웹툰 이미지를 분석하여 장면별로 효과음, 말풍선, 서사적 맥락을 정확히 추출하고, JSON 형식으로 구조화된 결과를 제공합니다."""
        return [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image_path"]},
                {"type": "text", "text": sample["query"]}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]

    def train(self) -> None:
        dataset = load_dataset('json', data_files=self.data_args.data_path)
        dataset = dataset['train'].train_test_split(test_size=1 - self.data_args.train_ratio, seed=42)
        train_dataset = [self.format_data(s) for s in dataset['train']]
        eval_dataset = [self.format_data(s) for s in dataset['test']]
        
        self.load_model_and_processor()

        data_collator = WeightedVLDataCollator(
            processor=self.processor,
            text_weight=self.data_args.text_weight,
            bbox_weight=self.data_args.bbox_weight
        )

        trainer = WeightedLossTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.training_args.early_stopping_patience)],
        )

        trainer.train()
        
        self.logger.info(f"Saving Full Model to {self.training_args.output_dir}")
        trainer.save_model(self.training_args.output_dir)
        self.processor.save_pretrained(self.training_args.output_dir)
        wandb.finish()

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    wandb.init(project=training_args.wandb_project, name=training_args.wandb_name)
    
    trainer = VLMTrainer(model_args, data_args, training_args)
    trainer.train()