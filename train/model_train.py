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
from accelerate import Accelerator, PartialState


@dataclass
class TrainingConfig:
    """학습 설정을 관리하는 데이터클래스"""
    
    # 데이터 경로
    data_path: str = '/workspace/Toonspace_VLM/data/grok_json_file/webtoon_balanced_training.json'
    output_dir: str = "ex_models/with_previous_toptoon_data_grok"
    
    # 모델 설정
    model_id: str = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"
    
    # 데이터 분할 비율
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 학습 하이퍼파라미터
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 3e-5
    max_grad_norm: float = 0.4
    warmup_ratio: float = 0.1
    
    # 프로세서 설정
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 960 * 28 * 28
    
    # 로깅 설정
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: int = 1000
    early_stopping_patience: int = 15
    
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
        # Accelerator 초기화
        self.accelerator = Accelerator()
        
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
            # if train_formatted and self.accelerator.is_main_process:
                # self.logger.info(f"샘플 데이터:\n{train_formatted[0]}")
            
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
                # device_map="auto",  # `accelerate`가 device_map을 자동 관리
                torch_dtype=torch.bfloat16
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_id,
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
            texts = [
                self.processor.apply_chat_template(example, tokenize=False) 
                for example in examples
            ]
            # print('===============================================\n\n')
            # print(examples[0]['conversation'])
            # print(type(examples[0]['conversation']))

            # print('===============================================\n\n')
            # print(examples[0]['conversation'][0])
            # print(type(examples[0]['conversation'][0]))

            # print('===============================================\n\n')
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
            
            # `accelerate`가 device를 자동으로 처리하므로, 여기서는 .to(device)를 제거
            return batch
        
        return collate_fn
    
    def train(self) -> None:
        """메인 학습 파이프라인 실행"""
        try:
            self.logger.info("VLM 파인튜닝 시작")
            
            # 메모리 정리
            self.clear_memory()
            
            # 데이터 로드
            train_dataset, eval_dataset, test_dataset_list = self.load_and_split_dataset()
            # train_dataset = Dataset.from_list(train_dataset_list)
            # eval_dataset = Dataset.from_list(eval_dataset_list)
            
            # 모델 및 옵티마이저 로드
            self.load_model_and_processor()
            optimizer = torch.optim.Adafactor(self.model.parameters(), lr=self.config.learning_rate)
            
            # 데이터 콜레이터 생성
            collate_fn = self.create_collate_fn()
            
            # 데이터 로더 생성
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            
            # `accelerate`로 모델, 옵티마이저, 데이터로더 준비
            self.model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, eval_dataloader
            )
            
            self.logger.info("학습 시작")
            
            # 커스텀 학습 루프
            for epoch in range(self.config.num_train_epochs):
                self.model.train()
                for step, batch in enumerate(train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # if self.accelerator.is_main_process and step % self.config.logging_steps == 0:
                        self.logger.info(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                        
                # 평가 루프
                self.model.eval()
                eval_loss = 0
                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = self.model(**batch)
                        eval_loss += outputs.loss.item()
                
                avg_eval_loss = eval_loss / len(eval_dataloader)
                self.logger.info(f"Epoch {epoch+1}, Avg Eval Loss: {avg_eval_loss:.4f}")
            
            # 메인 프로세스에서만 모델 저장
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                self.logger.info(f"모델 저장: {self.config.output_dir}")
                unwrapped_model.save_pretrained(self.config.output_dir)
                self.processor.save_pretrained(self.config.output_dir)
            
            self.logger.info("학습 완료")
            
        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {e}")
            raise
        finally:
            self.clear_memory()


def main():
    """메인 실행 함수"""
    try:
        # 설정 로드
        config = TrainingConfig()
        
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