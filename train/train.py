import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from trl import SFTConfig
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from accelerate import Accelerator

import wandb



@dataclass
class TrainingConfig:
    """학습 설정을 관리하는 데이터클래스"""
    
    # 데이터 경로
    data_path: str = '/workspace/Toonspace_VLM/data/grok_json_file/webtoon_balanced_training.json'
    output_dir: str = "ex_models/with_previous_toptoon_data_grok"
    
    # 모델 설정
    # model_id: str = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"
    model_id : str = '/workspace/Toonspace_VLM/ex_models/with_previous_toptoon_data_grok/checkpoint-10000'
    processor_id  : str = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"
    
    # 데이터 분할 비율
    train_ratio: float = 0.95
    eval_ratio: float = 0.0025
    test_ratio: float = 0.025
    
    # 학습 하이퍼파라미터
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    max_grad_norm: float = 0.4
    warmup_ratio: float = 0.1
    
    # 프로세서 설정
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 960 * 28 * 28
    
    # 로깅 설정
    logging_steps: int = 100
    eval_steps: int = 2000
    save_steps: int = 10000
    early_stopping_patience: int = 15

    #wandb 설정 추기 
    wandb_project_name: str = "Webtoon-vlm-finetuning"


    
    # 시스템 메시지
    system_message: str = field(default="""
    당신은 웹툰 이미지 분석 전문가입니다. 성인 웹툰 이미지를 분석하여 장면별로 효과음, 말풍선, 서사적 맥락을 정확히 추출하고, JSON 형식으로 구조화된 결과를 제공합니다. 모든 텍스트 요소(대사, 효과음, 나레이션)를 한국어로 추출하고, 캐릭터 관계와 상황 맥락을 세밀히 분석하며, 오해석을 최소화하십시오
    """)


def main():
    """메인 실행 함수"""
    # 환경 변수 설정 (RTX 이슈 해결용)
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_IB_DISABLE"] = "1"
    
    try:
        # 설정 로드
        config = TrainingConfig()
        

        # wandb 초기화
        wandb.init(project=config.wandb_project_name)



        # 학습 실행
        trainer = VLMTrainer(config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()