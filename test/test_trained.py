import torch
import json
import gc
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class EvaluationConfig:
    """평가 설정을 관리하는 데이터클래스"""
    checkpoint_path: str = "/workspace/Toonspace_VLM/ex_models/qwen25-7b-Webtoon_Analysis"
    batch_size: int = 8
    max_new_tokens: int = 128
    do_sample: bool = True
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 960 * 28 * 28
    train_ratio: float = 0.95
    eval_samples: int = 10
    test_samples: int = 10
    padding_side: str = "left"


@dataclass
class EvaluationMetrics:
    """평가 메트릭을 저장하는 데이터클래스"""
    bleu: float
    bleu_rt: float
    rouge: float
    f1: float


class WebtoonDataset(Dataset):
    """웹툰 VLM 학습용 데이터셋"""
    
    def __init__(self, data_list: List[Dict[str, Any]]):
        """
        Args:
            data_list: 각 요소가 다음 중 하나:
                - 원시 데이터: {"image_path": ..., "query": ..., "answer": ...}
                - 포맷된 데이터: [{"role": "system", ...}, {"role": "user", ...}, ...]
        """
        self.data = data_list
        print(f"데이터셋 초기화 완료: {len(data_list)}개 샘플")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class TaskTypeManager:
    """Task 타입 관련 기능을 관리하는 클래스"""
    
    SYSTEM_MESSAGES = {
        "scene_description": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰 이미지를 분석하여 장면의 상황, 맥락, 서사적 흐름을 정확하게 설명하세요.",
        "text_detection": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰 이미지에서 모든 텍스트 요소를 정확히 인식하고 위치 정보와 함께 추출하세요.",
        "terminology": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰에서 사용된 특별한 용어들을 식별하고 그 의미를 설명하세요.",
        "character_analysis": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰 캐릭터들을 분석하여 등장인물 정보와 감정 상태를 파악하세요.",
        "general": "당신은 웹툰 이미지 분석 전문가입니다. 성인 웹툰 이미지를 분석하여 장면별로 효과음, 말풍선, 서사적 맥락을 정확히 추출하고, JSON 형식으로 구조화된 결과를 제공합니다."
    }
    
    @classmethod
    def detect_task_type(cls, query: str) -> str:
        """쿼리에서 자동으로 task 타입을 감지"""
        query_lower = query.lower()
        
        if "장면" in query_lower or "상황" in query_lower or "묘사" in query_lower:
            return "scene_description"
        elif "텍스트" in query_lower or "말풍선" in query_lower or "인식" in query_lower:
            return "text_detection"
        elif "용어" in query_lower or "의미" in query_lower:
            return "terminology"
        elif "캐릭터" in query_lower or "등장인물" in query_lower or "감정" in query_lower:
            return "character_analysis"
        else:
            return "general"
    
    @classmethod
    def get_system_message(cls, task_type: str) -> str:
        """Task 타입에 따른 시스템 메시지 반환"""
        return cls.SYSTEM_MESSAGES.get(task_type, cls.SYSTEM_MESSAGES["general"])


class DataFormatter:
    """데이터 포맷팅을 담당하는 클래스"""
    
    @staticmethod
    def format_data(sample: Dict[str, Any], task_type: str = "general") -> List[Dict[str, Any]]:
        """
        웹툰 비전-언어 모델 학습을 위한 입력 데이터를 포맷팅
        
        Args:
            sample: 원시 데이터 샘플
            task_type: Task 유형
        
        Returns:
            포맷된 대화 리스트
        """
        # 이미지 로드 처리
        image = DataFormatter._load_image(sample)
        
        # 답변 처리
        answer = DataFormatter._process_answer(sample)
        
        # 시스템 메시지 생성
        system_message = TaskTypeManager.get_system_message(task_type)
        
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["query"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
    
    @staticmethod
    def _load_image(sample: Dict[str, Any]) -> Image.Image:
        """샘플에서 이미지를 로드"""
        if "image" in sample:
            return sample["image"]
        elif "image_path" in sample:
            return Image.open(sample["image_path"]).convert("RGB")
        else:
            raise ValueError("이미지 정보가 없습니다. 'image' 또는 'image_path' 키가 필요합니다.")
    
    @staticmethod
    def _process_answer(sample: Dict[str, Any]) -> str:
        """답변 데이터를 처리"""
        answer = sample.get("answer", sample.get("label", ""))
        if isinstance(answer, dict):
            return json.dumps(answer, ensure_ascii=False, indent=2)
        return str(answer)


class MemoryManager:
    """GPU 메모리 관리를 담당하는 클래스"""
    
    @staticmethod
    def clear_memory():
        """안전한 GPU 메모리 정리"""
        gc.collect()
        
        if torch.cuda.is_available():
            try:
                if torch.cuda.memory_allocated() > 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU allocated memory: {allocated:.2f} GB")
                print(f"GPU reserved memory: {reserved:.2f} GB")
                
            except RuntimeError as e:
                print(f"CUDA 동기화 중 오류 발생: {e}")
                print("GPU 메모리 정리를 건너뜁니다.")
        else:
            print("CUDA를 사용할 수 없습니다.")
        
        gc.collect()


class MetricsCalculator:
    """평가 메트릭 계산을 담당하는 클래스"""
    
    @staticmethod
    def calculate_bleu(reference: str, prediction: str) -> float:
        """BLEU score 계산"""
        smooth = SmoothingFunction().method1
        return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)
    
    @staticmethod
    def calculate_bleu_rt(reference: str, prediction: str) -> float:
        """BLEU-RT score 계산 (Brevity Penalty 조정 적용)"""
        candidate = prediction.split()
        ref_tokens = reference.split()
        candidate_len = len(candidate)
        reference_len = len(ref_tokens)
        
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], candidate, smoothing_function=smooth)
        
        if candidate_len > reference_len:
            bp = 1.0
        else:
            bp = np.exp(1 - (reference_len / candidate_len))
        
        return bleu * bp
    
    @staticmethod
    def calculate_rouge(reference: str, prediction: str) -> float:
        """ROUGE score 계산"""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rouge1'].fmeasure
    
    @staticmethod
    def calculate_f1(reference: str, prediction: str) -> float:
        """F1 score 계산 (단어 단위)"""
        reference_tokens = set(reference.split())
        prediction_tokens = set(prediction.split())
        common = reference_tokens & prediction_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(prediction_tokens) if prediction_tokens else 0
        recall = len(common) / len(reference_tokens) if reference_tokens else 0
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    @classmethod
    def get_all_metrics(cls, reference: str, prediction: str) -> EvaluationMetrics:
        """모든 메트릭을 계산하여 반환"""
        return EvaluationMetrics(
            bleu=cls.calculate_bleu(reference, prediction),
            bleu_rt=cls.calculate_bleu_rt(reference, prediction),
            rouge=cls.calculate_rouge(reference, prediction),
            f1=cls.calculate_f1(reference, prediction)
        )


class DataLoader:
    """데이터 로딩을 담당하는 클래스"""
    
    @staticmethod
    def load_webtoon_data(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """웹툰 JSON 데이터를 로드하고 처리"""
        json_path = Path(json_path)
        print(f"JSON 파일 로딩 중: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"로드된 데이터 수: {len(data)}")
            if data:
                print("첫 번째 데이터 키들:", list(data[0].keys()))
            
            return data
            
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {json_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            raise
        except Exception as e:
            print(f"데이터 로딩 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def split_data(data: List[Dict], config: EvaluationConfig) -> Tuple[List, List, List]:
        """데이터를 train/eval/test로 분할"""
        total_len = len(data)
        print(f"전체 데이터 수: {total_len}")
        
        if total_len < 10:
            print("경고: 데이터가 너무 적습니다. 최소 10개 이상의 샘플이 필요합니다.")
        
        train_end = int(config.train_ratio * total_len)
        eval_end = min(train_end + config.eval_samples, total_len)
        test_end = min(eval_end + config.test_samples, total_len)
        
        train_dataset = data[:train_end]
        eval_dataset = data[train_end:eval_end] if eval_end > train_end else []
        test_dataset = data[eval_end:test_end] if test_end > eval_end else []
        
        print(f"학습 데이터: {len(train_dataset)}개")
        print(f"검증 데이터: {len(eval_dataset)}개")
        print(f"테스트 데이터: {len(test_dataset)}개")
        
        return train_dataset, eval_dataset, test_dataset


class WebtoonVLMEvaluator:
    """웹툰 VLM 평가를 담당하는 메인 클래스"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._load_model_and_processor()
    
    def _load_model_and_processor(self):
        """모델과 프로세서를 로드"""
        print(f"모델 로딩 중: {self.config.checkpoint_path}")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.checkpoint_path,
            device_map="auto",
            torch_dtype='auto'
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.checkpoint_path,
            padding_side=self.config.padding_side,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels
        )
        
        self.model.eval()
        print("모델 및 프로세서 로딩 완료")
    
    def evaluate_batch(self, batch: List[List[Dict]]) -> Tuple[List[str], List[str]]:
        """배치 단위로 모델 평가"""
        texts, image_inputs_list, references = [], [], []
        
        for data in batch:
            references.append(data[2]['content'][0]['text'])
            messages = [data[0], data[1]]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            texts.append(text)
            image_inputs_list.append(image_inputs)
        
        # 배치 처리
        inputs = self.processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.max_new_tokens, 
                do_sample=self.config.do_sample
            )
        
        # 디코딩
        output_texts = []
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
            generated_ids_trimmed = out_ids[len(in_ids):]
            decoded = self.processor.decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            output_texts.append(decoded)
        
        return output_texts, references
    
    def evaluate_dataset(self, dataset: WebtoonDataset) -> Dict[str, float]:
        """전체 데이터셋에 대한 평가 수행"""
        all_metrics = []
        
        for i in tqdm(range(0, len(dataset), self.config.batch_size), desc="Evaluating"):
            batch = dataset[i : i + self.config.batch_size]
            batch_generated, batch_references = self.evaluate_batch(batch)
            
            for generated, reference in zip(batch_generated, batch_references):
                metrics = MetricsCalculator.get_all_metrics(reference, generated)
                all_metrics.append(metrics)
        
        # 평균 계산
        if not all_metrics:
            return {}
        
        avg_metrics = {
            'bleu': sum(m.bleu for m in all_metrics) / len(all_metrics),
            'bleu_rt': sum(m.bleu_rt for m in all_metrics) / len(all_metrics),
            'rouge': sum(m.rouge for m in all_metrics) / len(all_metrics),
            'f1': sum(m.f1 for m in all_metrics) / len(all_metrics)
        }
        
        return avg_metrics


def main():
    """메인 실행 함수"""
    # 설정 초기화
    config = EvaluationConfig()
    
    # 데이터 로드
    webtoon_path = '/workspace/Toonspace_VLM/data/json_file/webtoon_balanced_training.json'
    raw_data = DataLoader.load_webtoon_data(webtoon_path)
    
    # 데이터 분할
    train_data, eval_data, test_data = DataLoader.split_data(raw_data, config)
    
    if not test_data:
        print("테스트 데이터가 없습니다.")
        return
    
    # 데이터 포맷팅
    print("데이터 포맷팅 중...")
    task_type = "auto"  # 또는 특정 task type 지정
    
    formatted_test_data = []
    for sample in test_data:
        if task_type == "auto":
            detected_task = TaskTypeManager.detect_task_type(sample["query"])
        else:
            detected_task = task_type
        
        formatted_sample = DataFormatter.format_data(sample, detected_task)
        formatted_test_data.append(formatted_sample)
    
    # 데이터셋 생성
    test_dataset = WebtoonDataset(formatted_test_data)
    
    # 평가자 초기화 및 평가 실행
    evaluator = WebtoonVLMEvaluator(config)
    
    try:
        results = evaluator.evaluate_dataset(test_dataset)
        
        print("\n=== 평가 결과 ===")
        for metric_name, value in results.items():
            print(f"{metric_name.upper()}: {value:.4f}")
            
    finally:
        # 메모리 정리
        print('===================!!Memory Clearning!!===================')
        MemoryManager.clear_memory()


if __name__ == "__main__":
    main()