"""
차트/웹툰 질문 답변을 위한 Vision Language Model 3단계 파이프라인 학습 스크립트

Step 1: Adapter (Merger) Only - 시각 정보를 언어 모델에 맞추는 정렬 학습
Step 2: Adapter + Decoder (LLM) - 시각 인코더는 고정하고 언어 모델과 연결부 학습
Step 3: Full Fine-tuning - 전체 모델 미세 조정

작성자: 정원렬 (Refactored by Assistant)
Year: 2025
"""

import gc
import logging

import os
os.environ["CODECARBON_DISABLED"] = "true"
os.environ["CODECARBON_LOG_LEVEL"] = "error"

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn 
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info
from trl import SFTConfig
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Qwen3VLForConditionalGeneration, 
    Trainer,
    TrainingArguments,
)
import wandb

# 사용자 커스텀 모듈 임포트 (경로가 맞아야 함)
from toonVLM.weight_utils import WeightedVLDataCollator, WeightedLossTrainer


@dataclass
class TrainingConfig:
    """학습 설정을 관리하는 데이터클래스"""
    
    # 데이터 경로
    data_path: str = '/workspace/Toonspace_VLM/data/ocr_description/total(bbox_normal)_ocr_dataset_2F.json'
    base_output_dir: str = "ex_models/qwen_3_OCR_3stage_pipeline_10:1"
    
    # 모델 설정
    model_id: str = 'huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated'
    processor_id: str = 'huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated'
    
    # 데이터 분할 비율
    train_ratio: float = 0.95
    eval_ratio: float = 0.025
    
    # 학습 하이퍼파라미터 (공통)
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.4
    warmup_ratio: float = 0.03
    
    # 가중치 설정
    text_weight: float = 1.0
    bbox_weight: float = 0.1

    # 이미지 설정
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 960 * 28 * 28
    
    # 로깅 설정
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 1000
    early_stopping_patience: int = 10
    
    # WandB
    wandb_project: str = "qwen3-Webtoon-vlm-OCR-3stage"
    
    system_message: str = field(default="""
    당신은 웹툰 이미지 분석 전문가입니다. 성인 웹툰 이미지를 분석하여 장면별로 효과음, 말풍선, 서사적 맥락을 정확히 추출하고, JSON 형식으로 구조화된 결과를 제공합니다. 모든 텍스트 요소(대사, 효과음, 나레이션)를 한국어로 추출하고, 캐릭터 관계와 상황 맥락을 세밀히 분석하며, 오해석을 최소화하십시오
    """)

    # --- 단계별 설정 (Epochs & LR) ---
    step1_epochs: int = 1
    step1_lr: float = 1e-3  # Adapter만 학습하므로 높은 LR
    
    step2_epochs: int = 3
    step2_lr: float = 2e-5  # LLM 학습 시작
    
    step3_epochs: int = 5
    step3_lr: float = 5e-6  # 전체 미세조정은 낮은 LR


class VLMTrainer:
    """Vision Language Model 파인튜닝을 위한 메인 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
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
    
    def clear_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def format_data(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.config.system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image_path"]},
                {"type": "text", "text": sample["query"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]
    
    def load_dataset(self):
        """데이터셋 로드 및 분할 (한 번만 실행)"""
        dataset = load_dataset('json', data_files=self.config.data_path)
        dataset['train'] = dataset["train"].shuffle(seed=42)
        total_size = len(dataset["train"])
        
        train_size = int(self.config.train_ratio * total_size)
        eval_size = int(self.config.eval_ratio * total_size)
        
        train_ds = dataset["train"].select(range(0, train_size))
        eval_ds = dataset["train"].select(range(train_size, train_size + eval_size))
        
        # 포맷팅 적용
        train_formatted = [self.format_data(sample) for sample in train_ds]
        eval_formatted = [self.format_data(sample) for sample in eval_ds]
        
        return train_formatted, eval_formatted

    def load_model(self, model_path: str):
        """모델 로드"""
        self.logger.info(f"모델 로드 중... 경로: {model_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2" # 가능하다면 Flash Attention 사용 권장
        )
        # Processor는 처음에 한 번만 로드해도 됨 (단, 저장된거 불러올때 주의)
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.config.processor_id,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels
            )

    def configure_freezing(self, step: int):
        """
        단계별 모델 동결/해제 설정 (핵심 로직)
        
        Qwen2.5/3-VL 구조 가정:
        - visual (Vision Encoder)
            - merger (Adapter/Projector)
            - blocks (ViT Layers)
        - model (LLM Decoder)
        """
        self.logger.info(f"Step {step} Freezing 설정 적용 중...")
        
        # 전체 파라미터 수 확인용
        trainable_params = 0
        all_param = 0
        
        for name, param in self.model.named_parameters():
            all_param += param.numel()
            param.requires_grad = False  # 일단 모두 False로 초기화

            # --- Step 1: Adapter (Merger) Only ---
            if step == 1:
                # 'merger'가 포함된 레이어만 학습 (Visual Projector)
                if "merger" in name or "projector" in name: 
                    param.requires_grad = True
            
            # --- Step 2: Adapter + Decoder (LLM) ---
            elif step == 2:
                # Vision Encoder 부분은 제외하고 학습
                # 보통 'visual' 안에 merger가 포함되므로 주의 필요
                if "visual" in name:
                    # Visual 내부에서도 merger는 학습해야 함
                    if "merger" in name or "projector" in name:
                        param.requires_grad = True
                    else:
                        # 순수 Vision Encoder (Blocks 등)는 동결
                        param.requires_grad = False
                else:
                    # 'visual'이 이름에 없으면 LLM 파트이므로 학습 (model.layers 등)
                    param.requires_grad = True

            # --- Step 3: Full Fine-tuning ---
            elif step == 3:
                # 전체 학습
                param.requires_grad = True

            if param.requires_grad:
                trainable_params += param.numel()

        self.logger.info(
            f"Step {step} - Trainable params: {trainable_params} || "
            f"All params: {all_param} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )

    def train_stage(self, step: int, train_dataset, eval_dataset, resume_from_checkpoint=None):
        """특정 단계(Step) 학습 실행"""
        
        # 단계별 설정 가져오기
        if step == 1:
            epochs = self.config.step1_epochs
            lr = self.config.step1_lr
            stage_name = "step1_adapter_only"
        elif step == 2:
            epochs = self.config.step2_epochs
            lr = self.config.step2_lr
            stage_name = "step2_adapter_decoder"
        else:
            epochs = self.config.step3_epochs
            lr = self.config.step3_lr
            stage_name = "step3_full_finetune"

        output_dir = os.path.join(self.config.base_output_dir, stage_name)
        
        # 모델 동결 설정 적용
        self.configure_freezing(step)

        # Training Arguments 생성
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adafactor",
            lr_scheduler_type="cosine",
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            save_safetensors=True, 
            bf16=True,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            run_name=f"{self.config.wandb_project}-{stage_name}",
            ddp_find_unused_parameters=False,
            dataset_text_field="", # SFTConfig 필수값 더미
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "min_num_params": 0,
                "xla": False,
                "xla_fsdp_grad_ckpt": False,
                "limit_all_gathers": True,
                "use_orig_params": True, # 파라미터 동결시 True 필수 (FSDP 버그 방지)
                "sync_module_states": True,
            },
        )

        if step < 3:
            text_weight = 1
            bbox_weight = 1
        else:
            text_weight = self.config.text_weight
            bbox_weight = self.config.bbox_weight
        
        data_collator = WeightedVLDataCollator(
            processor=self.processor,
            text_weight=text_weight,
            bbox_weight=bbox_weight
        )

        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
        )

        self.logger.info(f"=== {stage_name} 학습 시작 ===")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 모델 저장
        self.logger.info(f"=== {stage_name} 학습 완료 및 저장 ===")
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        
        # 메모리 정리
        del trainer
        self.clear_memory()
        
        return output_dir  # 다음 단계에서 로드할 경로 반환

    def run_pipeline(self):
        """전체 3단계 파이프라인 실행"""
        # 1. 데이터 로드
        train_dataset, eval_dataset = self.load_dataset()
        
        # 2. 초기 모델 로드 (Base Model)
        current_model_path = self.config.model_id
        self.load_model(current_model_path)

        # ----------------------
        # Step 1: Adapter Only
        # ----------------------
        step1_output_dir = self.train_stage(1, train_dataset, eval_dataset)
        
        # ----------------------
        # Step 2: Adapter + Decoder
        # ----------------------
        # Step 1 결과 모델 로드 (Trainer가 메모리에 갖고 있지만, 확실한 리셋을 위해 다시 로드 권장)
        # FSDP 사용 시 모델 구조 변경(Freeze/Unfreeze) 후 재로드가 안전함
        del self.model
        self.clear_memory()
        self.load_model(step1_output_dir) # Step 1에서 저장된 모델 로드
        
        step2_output_dir = self.train_stage(2, train_dataset, eval_dataset)

        # ----------------------
        # Step 3: Full Fine-tuning
        # ----------------------
        del self.model
        self.clear_memory()
        self.load_model(step2_output_dir) # Step 2에서 저장된 모델 로드
        
        final_output_dir = self.train_stage(3, train_dataset, eval_dataset)
        
        self.logger.info(f"모든 학습 단계 완료! 최종 모델 경로: {final_output_dir}")


def main():
    try:
        config = TrainingConfig()
        wandb.init(project=config.wandb_project, name="3stage_pipeline_run")
        
        trainer = VLMTrainer(config)
        trainer.run_pipeline()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()