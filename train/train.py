import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import requests
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from trl import SFTConfig
from transformers import (
    AutoProcessor,
)
from torch.cuda.amp import GradScaler

from accelerate import Accelerator
import wandb

from toonVLM.qwen_multi_head import Multihead_Toonspace



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


from torch.cuda.amp import GradScaler, autocast

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"
    
    print(f"디바이스: {device} | 모델: {model_name}\n")
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    # 모델을 float32 상태로 로드합니다. autocast가 타입을 관리합니다.
    model = Multihead_Toonspace(model_name).to(device)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    gt_ocr_text = "A cat is sitting on a couch."
    gt_desc_text = "A cat is resting on a blue and white striped couch."

    ocr_labels = tokenizer(gt_ocr_text, return_tensors='pt', padding='max_length', max_length=30, truncation=True).input_ids.to(device)
    desc_labels = tokenizer(gt_desc_text, return_tensors='pt', padding='max_length', max_length=30, truncation=True).input_ids.to(device)

    print("--- 훈련 (10 스텝) ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for step in range(35):
        optimizer.zero_grad()
    
        # BFloat16 사용을 명시하여 autocast 컨텍스트 실행
        # CPU 환경에서는 autocast가 비활성화됩니다.
        with autocast(enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
            total_loss = model(
                image=image,
                ocr_labels=ocr_labels,
                desc_labels=desc_labels,
                processor=processor,
                tokenizer=tokenizer
            )

        print(f"스텝 {step+1} | 계산된 총 손실 (Total Loss): {total_loss.item():.4f}")
        
        if not torch.isnan(total_loss):
            # FIX: GradScaler 관련 코드 모두 제거
            # 1. total_loss를 직접 backward() 호출
            total_loss.backward()
            
            # 2. 그래디언트 클리핑 (선택사항이지만 안정성을 위해 권장)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 3. 옵티마이저 스텝 바로 실행
            optimizer.step()
        else:
            print("손실이 nan이므로 스텝을 건너뜁니다.")
            break
            
    print("\n역전파 및 옵티마이저 스텝 완료.\n")
    
    print("--- 추론 (생성) ---")
    with autocast(enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
        generated_ocr, generated_desc = model.generate(
            image=image,
            tokenizer=tokenizer,
            processor=processor,
            max_length=30
        )
    
    print(f"OCR 헤드 생성 결과 (Instruction-based): '{generated_ocr}'")
    print(f"설명 헤드 생성 결과 (Instruction-based): '{generated_desc}'")

if __name__ == "__main__":
    main()

