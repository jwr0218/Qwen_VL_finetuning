"""
차트 질문 답변을 위한 Vision Language Model 파인튜닝 스크립트 (리팩토링 버전)

이 모듈은 차트 분석 작업을 위한 Qwen2.5-VL 모델의 파인튜닝을 구현합니다.
전체 파인튜닝(Full Fine-tuning)을 사용하여 차트 이미지와 
관련 질문 및 답변을 처리합니다.

작성자: 정원렬
Year: 2025
Month: 8
Day: 27
"""

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


class VLMTrainer:
    """Vision Language Model 파인튜닝을 위한 메인 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.model = None
        self.processor = None
        
    def setup_logging(self) -> None:
        """로깅 시스템 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_environment(self) -> None:
        """환경 변수 및 분산 훈련 설정"""
        # NCCL 및 분산 훈련 설정
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_P2P_DISABLE", "0")  # P2P 통신 유지
        os.environ.setdefault("NCCL_IB_DISABLE", "0")   # InfiniBand 유지
        
        # 분산 훈련 안정성 개선
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
        
        # FSDP 관련 설정
        os.environ.setdefault("FSDP_CPU_OFFLOAD", "0")
        
        self.logger.info("환경 설정 완료")


    def clear_memory(self) -> None:
        """안전한 GPU 메모리 정리"""
        gc.collect()
        
        if torch.cuda.is_available():
            try:
                if torch.cuda.memory_allocated() > 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb = torch.cuda.memory_reserved() / 1024**3
                
                self.logger.info(f"GPU allocated memory: {allocated_gb:.2f} GB")
                self.logger.info(f"GPU reserved memory: {reserved_gb:.2f} GB")
                
            except RuntimeError as e:
                self.logger.warning(f"CUDA 동기화 중 오류 발생: {e}")
                self.logger.warning("GPU 메모리 정리를 건너뜁니다.")
        else:
            self.logger.warning("CUDA를 사용할 수 없습니다.")
        
        gc.collect()
    
    def format_data(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        비전-언어 모델 학습을 위한 입력 데이터 포맷팅
        
        Args:
            sample: 원시 데이터 샘플
            
        Returns:
            포맷된 대화 형태의 데이터
        """
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": sample["query"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
    
    def load_and_split_dataset(self) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        데이터셋 로드 및 분할
        
        Returns:
            train_dataset, eval_dataset, test_dataset
        """
        try:
            if not Path(self.config.data_path).exists():
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.config.data_path}")
            
            dataset = load_dataset('json', data_files=self.config.data_path)
            dataset['train'] = dataset["train"].shuffle(seed=42)
            total_size = len(dataset["train"])
            
            # 데이터셋 분할 인덱스 계산
            train_size = int(self.config.train_ratio * total_size)
            eval_size = int(self.config.eval_ratio * total_size)
            
            # 데이터셋 분할
            train_dataset = dataset["train"].select(range(0, train_size))
            eval_dataset = dataset["train"].select(range(train_size, train_size + eval_size))
            test_dataset = dataset["train"].select(range(train_size + eval_size, train_size + eval_size * 2))
            
            # 데이터 포맷팅
            train_formatted = [self.format_data(sample) for sample in train_dataset]
            eval_formatted = [self.format_data(sample) for sample in eval_dataset]
            test_formatted = [self.format_data(sample) for sample in test_dataset]
            
            self.logger.info(f"데이터셋 로드 완료:")
            self.logger.info(f"  - 학습 데이터: {len(train_formatted)}개")
            self.logger.info(f"  - 검증 데이터: {len(eval_formatted)}개") 
            self.logger.info(f"  - 테스트 데이터: {len(test_formatted)}개")
            
            # 샘플 데이터 출력
            # if train_formatted:
            #     self.logger.info(f"샘플 데이터:\n{train_formatted[0]}")
            
            return train_formatted, eval_formatted, test_formatted
            
        except Exception as e:
            self.logger.error(f"데이터셋 로드 중 오류 발생: {e}")
            raise
    
    def load_model_and_processor(self) -> None:
        """모델 및 프로세서 로드"""
        try:
            self.logger.info(f"모델 로드 시작: {self.config.model_id}")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_id,
                # device_map="cuda",
                torch_dtype=torch.bfloat16, 
                trust_remote_code = True
            )

            self.processor = AutoProcessor.from_pretrained(
                self.config.processor_id,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels
            )
            
            self.logger.info("모델 및 프로세서 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def create_collate_fn(self):
        """데이터 콜레이션 함수 생성"""
        def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
            """
            비전-언어 모델 학습을 위한 사용자 정의 데이터 콜레이션 함수
            
            Args:
                examples: 포맷된 대화 예제 목록
                
            Returns:
                배치 딕셔너리 (input_ids, attention_mask, pixel_values, labels)
            """
            # print(examples)
            # exit()
            texts = [
                self.processor.apply_chat_template(example, tokenize=False) 
                for example in examples
            ]

            image_inputs = [process_vision_info(example)[0] for example in examples]
            
            batch = self.processor(
                text=texts, 
                images=image_inputs, 
                return_tensors="pt", 
                padding=True
            )
            
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            
            return batch
        
        return collate_fn
    
    def create_training_args(self) -> SFTConfig:
        """학습 설정 생성 (FSDP 활성화)"""
        return SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adafactor",
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="constant",
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=self.config.save_steps,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            # dataloader_pin_memory=False,
            save_safetensors=True, 
            bf16=True,
            fp16=False,       
            bf16_full_eval=True,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            ddp_find_unused_parameters=False,  # FSDP 사용 시 False 권장
            dataloader_num_workers=1,
            report_to=["wandb"],
            run_name=self.config.wandb_project_name,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            local_rank=-1,
            # FSDP 활성화
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "min_num_params": 0,
                "xla": False,
                "xla_fsdp_grad_ckpt": False,
                "backward_prefetch": "backward_pre",
                "forward_prefetch": False,
                "limit_all_gathers": True,
                "use_orig_params": False,
                "cpu_offload": False,
                "sync_module_states": True,  # 중요: 모듈 상태 동기화
            },
        )
    
    def train(self) -> None:
        """메인 학습 파이프라인 실행"""
        try:
            self.logger.info("VLM 파인튜닝 시작")
            
            # 메모리 정리
            self.clear_memory()
            
            # 데이터 로드
            train_dataset, eval_dataset, test_dataset = self.load_and_split_dataset()
            
            # 모델 로드
            self.load_model_and_processor()
            
            # 학습 설정
            training_args = self.create_training_args()
            collate_fn = self.create_collate_fn()
            
            # Trainer 설정
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collate_fn,
                tokenizer=self.processor.tokenizer,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )],
            )
            
            # 학습 실행
            self.logger.info("학습 시작")
            trainer.train()
            
            # 모델 저장
            self.logger.info(f"모델 저장: {self.config.output_dir}")
            trainer.save_model(self.config.output_dir)
            self.processor.save_pretrained(self.config.output_dir)
            
            self.logger.info("학습 완료")
            
        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {e}")
            raise
        finally:
            self.clear_memory()

            wandb.finish()


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