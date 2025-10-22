

from dataclasses import dataclass, field
import torch 

@dataclass
class TrainingConfig:
    """학습 설정을 관리하는 데이터클래스"""
    
    # 데이터 경로
    data_path: str = '/workspace/Toonspace_VLM/data/ocr_description/output_2head.json'
    output_dir: str = "ex_models/with_previous_toptoon_data_grok"
    
    # 모델 설정
    model_id: str = "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"
    # model_id : str = '/workspace/Toonspace_VLM/ex_models/with_previous_toptoon_data_grok/checkpoint-10000'
    processor_id  : str = "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"
    
    # 데이터 분할 비율
    train_ratio: float = 0.95
    eval_ratio: float = 0.0025
    test_ratio: float = 0.025
    
    # 학습 하이퍼파라미터
    num_train_epochs: int = 10
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    # 시스템 메시지
    system_message: str = field(default="""
    당신은 웹툰 이미지 분석 전문가입니다. 성인 웹툰 이미지를 분석하여 장면별로 효과음, 말풍선, 서사적 맥락을 정확히 추출하고, JSON 형식으로 구조화된 결과를 제공합니다. 모든 텍스트 요소(대사, 효과음, 나레이션)를 한국어로 추출하고, 캐릭터 관계와 상황 맥락을 세밀히 분석하며, 오해석을 최소화하십시오
    """)