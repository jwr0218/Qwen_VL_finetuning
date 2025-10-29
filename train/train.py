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



from torch.cuda.amp import GradScaler, autocast

from toonVLM.TrainingConfig import TrainingConfig




def main():
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"
    
    print(f"디바이스: {device} | 모델: {config.model_id}\n")
    
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    # 모델 불러오기 & Optimizer 불러오기 
    if config.multihead : 
        from toonVLM.qwen_multi_head import Multihead_Toonspace
        from toonVLM.train_utils_multihead import VLMTrainer

        model = Multihead_Toonspace(config.model_id).to(device)
    else: 
        from transformers import Qwen2_5_VLForConditionalGeneration
        from toonVLM.train_utils import VLMTrainer
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_id,
            # device_map="cuda",
            torch_dtype=torch.bfloat16, 
            trust_remote_code = True
        )

        if config.apply_back_extension:
            from toonVLM.back_expansion import apply_back_extension_with_freeze
            print("Back Extension (Zero-Init Deepening)을 적용합니다...")
            
            # 위에서 정의한 apply_back_extension 함수 호출
            model = apply_back_extension_with_freeze(
                original_model=model,
                target_layers=config.target_layers
            )
            
            print(
                f"모델 Deepening 완료. 새 레이어 수: {model.config.num_hidden_layers}"
                )

        processor = AutoProcessor.from_pretrained(
            config.processor_id,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels
        )

        
    model.train()
    optimizer = torch.optim.Adafactor(model.parameters(), lr=1e-5)

    
    # 학습 실행
    trainer = VLMTrainer(config,model,processor,optimizer)
    
    trainer.train()
if __name__ == "__main__":
    main()

