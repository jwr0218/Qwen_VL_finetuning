"""
Vision Language Model 예측 및 테스트 스크립트 (웹툰 순차 추론 버전)

이 모듈은 파인튜닝된 Qwen2.5-VL 모델의 예측 및 평가를 구현합니다.
웹툰 이미지를 순차적으로 분석하며 이전 컨텍스트를 누적하여 처리합니다.

작성자: Assistant
Year: 2025
Month: 9
Day: 17
"""

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import re

import numpy as np
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from PIL import Image

# 평가 지표 관련 라이브러리
from rouge import Rouge
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize, download
import evaluate
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# NLTK 데이터 다운로드
try:
    download('punkt')
    download('wordnet')
except:
    pass


@dataclass
class TestConfig:
    """테스트 설정을 관리하는 데이터클래스"""
    
    # 데이터 경로
    test_data_path: str = '/workspace/Toonspace_VLM/data/grok_json_file/webtoon_balanced_test.json'
    
    # 모델 경로 (파인튜닝된 모델)
    model_path: str = '/workspace/Toonspace_VLM/ex_models/with_previous_toptoon_data_grok/checkpoint-10000'
    
    # 결과 저장 경로
    output_dir: str = 'evaluation_results'
    predictions_file: str = 'predictions.json'
    metrics_file: str = 'evaluation_metrics.json'
    
    # 배치 설정
    batch_size: int = 1
    max_samples: Optional[int] = None  # None이면 전체 데이터 사용
    
    # 생성 설정
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # 프로세서 설정
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 960 * 28 * 28
    
    # 평가 설정
    compute_bertscore: bool = True
    compute_json_metrics: bool = True  # JSON 구조 평가
    
    # 시스템 메시지
    system_message: str = field(default="""
    당신은 웹툰 이미지 분석 전문가입니다. 성인 웹툰 이미지를 분석하여 장면별로 효과음, 말풍선, 서사적 맥락을 정확히 추출하고, JSON 형식으로 구조화된 결과를 제공합니다. 모든 텍스트 요소(대사, 효과음, 나레이션)를 한국어로 추출하고, 캐릭터 관계와 상황 맥락을 세밀히 분석하며, 오해석을 최소화하십시오
    """)


@dataclass
class WebtoonPredictConfig:
    """웹툰 폴더 예측 설정을 관리하는 데이터클래스"""
    
    # 모델 경로
    model_path: str = '/workspace/Toonspace_VLM/ex_models/with_previous_toptoon_data_grok/checkpoint-10000'
    
    # 원본 모델 ID (프로세서 로드용)
    base_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"  # 또는 "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"
    
    # 입력 폴더 경로
    image_folder: str = '/workspace/Toonspace_VLM/webtoon_images'
    
    # 결과 저장 경로
    output_dir: str = 'webtoon_predictions'
    output_file: str = 'sequential_predictions.json'
    
    # 생성 설정
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # 프로세서 설정
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 960 * 28 * 28
    
    # 컨텍스트 설정
    use_previous_context: bool = True  # 이전 추론 결과를 다음 추론에 사용
    max_context_length: int = 2000  # 최대 컨텍스트 길이 (토큰 수)
    context_summary_length: int = 500  # 컨텍스트 요약 길이
    
    # 이미지 파일 확장자
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.webp'])
    
    # 프롬프트 템플릿
    base_prompt: str = "이 웹툰 이미지를 분석하여 대사, 효과음, 캐릭터 행동, 감정을 JSON 형식으로 추출해주세요."
    
    context_prompt_template: str = """이전 장면 요약:
{previous_context}

현재 이미지를 위의 컨텍스트를 고려하여 분석해주세요. 이 웹툰 이미지의 대사, 효과음, 캐릭터 행동, 감정을 JSON 형식으로 추출해주세요."""
    
    # 시스템 메시지
    system_message: str = field(default="""
    당신은 웹툰 이미지 분석 전문가입니다. 웹툰의 연속된 장면을 분석하여 스토리의 흐름을 이해하고, 
    각 장면의 대사, 효과음, 캐릭터의 행동과 감정, 서사적 맥락을 정확히 추출합니다. 
    이전 장면의 맥락을 고려하여 현재 장면을 더 정확하게 해석하고, 
    모든 결과를 구조화된 JSON 형식으로 제공합니다.
    """)


class WebtoonSequentialPredictor:
    """웹툰 이미지 순차 예측을 위한 클래스"""
    
    def __init__(self, config: WebtoonPredictConfig):
        self.config = config
        self.setup_logging()
        self.model = None
        self.processor = None
        self.context_history = []  # 이전 추론 결과 저장
        
    def setup_logging(self) -> None:
        """로깅 시스템 설정"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_dir}/webtoon_prediction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def clear_memory(self) -> None:
        """GPU 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_model_and_processor(self) -> None:
        """파인튜닝된 모델 및 프로세서 로드"""
        try:
            self.logger.info(f"모델 로드 시작: {self.config.model_path}")
            
            # 로컬 경로 확인 및 모델 로드
            if os.path.exists(self.config.model_path):
                self.logger.info("로컬 모델 파일 감지됨")
                
                # 모델 로드
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.logger.info(f"원본 모델({self.config.base_model_id})에서 프로세서 로드")
                self.processor = AutoProcessor.from_pretrained(
                    self.config.base_model_id,
                    min_pixels=self.config.min_pixels,
                    max_pixels=self.config.max_pixels,
                    trust_remote_code=True
                )

            
            self.model.eval()
            self.logger.info("모델 및 프로세서 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            self.logger.error("체크포인트 경로를 확인하거나 base_model_id를 올바르게 설정해주세요.")
            raise
    
    def get_sorted_image_files(self, folder_path: str) -> List[Path]:
        """폴더에서 이미지 파일을 정렬된 순서로 가져오기"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
        
        image_files = []
        for ext in self.config.image_extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        # 파일명 기준 정렬 (숫자가 포함된 경우 자연스러운 정렬)
        def natural_sort_key(path):
            """자연스러운 정렬을 위한 키 함수"""
            text = path.name
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
        
        image_files.sort(key=natural_sort_key)
        
        self.logger.info(f"총 {len(image_files)}개의 이미지 파일 발견")
        return image_files
    
    def extract_key_info_from_prediction(self, prediction: str) -> str:
        """예측 결과에서 핵심 정보 추출 및 요약"""
        try:
            # JSON 파싱 시도
            pred_json = json.loads(prediction)
            
            summary_parts = []
            
            # 대사 추출
            if 'dialogues' in pred_json:
                dialogues = pred_json['dialogues']
                if isinstance(dialogues, list) and dialogues:
                    dialogue_text = ', '.join([d.get('text', '') for d in dialogues[:3]])
                    summary_parts.append(f"대사: {dialogue_text}")
            
            # 캐릭터 행동 추출
            if 'characters' in pred_json:
                characters = pred_json['characters']
                if isinstance(characters, list) and characters:
                    char_actions = ', '.join([c.get('action', '') for c in characters[:2] if 'action' in c])
                    if char_actions:
                        summary_parts.append(f"행동: {char_actions}")
            
            # 감정 추출
            if 'emotions' in pred_json:
                emotions = pred_json['emotions']
                if isinstance(emotions, list):
                    emotion_text = ', '.join(emotions[:3])
                    summary_parts.append(f"감정: {emotion_text}")
            
            # 장면 설명 추출
            if 'scene_description' in pred_json:
                scene = pred_json['scene_description'][:100]
                summary_parts.append(f"장면: {scene}")
            
            return ' | '.join(summary_parts)
            
        except json.JSONDecodeError:
            # JSON이 아닌 경우 텍스트 요약
            return prediction[:self.config.context_summary_length]
        except Exception as e:
            self.logger.debug(f"요약 추출 중 오류: {e}")
            return prediction[:self.config.context_summary_length]
    
    def build_context_prompt(self, base_query: str) -> str:
        """이전 컨텍스트를 포함한 프롬프트 생성"""
        if not self.config.use_previous_context or not self.context_history:
            return base_query
        
        # 최근 3개 정도의 컨텍스트만 사용 (메모리 효율성)
        recent_context = self.context_history[-3:]
        context_summary = "\n".join([f"장면 {i+1}: {ctx}" for i, ctx in enumerate(recent_context)])
        
        # 컨텍스트 길이 제한
        if len(context_summary) > self.config.max_context_length:
            context_summary = context_summary[-self.config.max_context_length:]
        
        return self.config.context_prompt_template.format(
            previous_context=context_summary
        )
    
    @torch.no_grad()
    def predict_single_image(self, image_path: Path, with_context: bool = True) -> Dict[str, Any]:
        """단일 이미지 예측"""
        # 프롬프트 생성
        if with_context and self.context_history:
            query = self.build_context_prompt(self.config.base_prompt)
        else:
            query = self.config.base_prompt
        
        # 메시지 포맷팅
        # 메시지 포맷팅
        query = '이 웹툰 이미지에서 모든 텍스트를 인식하고 추출해주세요.'
        messages = [
            {
                "role": "system",
                "content": self.config.system_message,  # 리스트가 아닌 문자열로!
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": query},
                ],
            },
        ]
        
        # 텍스트 생성
        # print(messages)
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 생성
        start_time = time.time()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
        )
        generation_time = time.time() - start_time
        
        # 생성된 토큰만 추출
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 디코딩
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'query': query,
            'prediction': output_text,
            'generation_time': generation_time,
            'used_context': with_context and bool(self.context_history),
            'context_length': len(self.context_history)
        }
            
    
    def predict_folder(self, folder_path: str = None) -> List[Dict[str, Any]]:
        """폴더 내 모든 이미지를 순차적으로 예측"""
        if folder_path is None:
            folder_path = self.config.image_folder
        
        self.logger.info(f"폴더 예측 시작: {folder_path}")
        
        # 이미지 파일 목록 가져오기
        image_files = self.get_sorted_image_files(folder_path)
        if not image_files:
            self.logger.warning("이미지 파일을 찾을 수 없습니다.")
            return []
        
        # 모델 로드
        self.load_model_and_processor()
        
        # 결과 저장
        all_results = []
        self.context_history = []  # 컨텍스트 초기화
        
        # 순차적으로 이미지 처리
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            # self.logger.info(f"처리 중 ({idx+1}/{len(image_files)}): {image_path.name}")
            
            # 현재 이미지 예측
            result = self.predict_single_image(
                image_path, 
                with_context=(idx > 0 and self.config.use_previous_context)
            )
            
            # 결과에 순서 정보 추가
            result['sequence_number'] = idx + 1
            result['total_images'] = len(image_files)
            
            # 컨텍스트 업데이트
            if 'prediction' in result and self.config.use_previous_context:
                context_summary = self.extract_key_info_from_prediction(result['prediction'])
                self.context_history.append(context_summary)
                
                # 컨텍스트 히스토리 크기 제한 (메모리 관리)
                if len(self.context_history) > 5:
                    self.context_history.pop(0)
            
            all_results.append(result)
            
            # 주기적으로 메모리 정리
            if (idx + 1) % 10 == 0:
                self.clear_memory()
                self.logger.info(f"메모리 정리 완료 ({idx+1}/{len(image_files)})")
        
        # 결과 저장
        self.save_results(all_results, folder_path)
        
        self.logger.info(f"예측 완료: 총 {len(all_results)}개 이미지 처리")
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]], source_folder: str) -> None:
        """결과를 파일로 저장"""
        # 메타데이터 추가
        output_data = {
            'metadata': {
                'source_folder': source_folder,
                'model_path': self.config.model_path,
                'total_images': len(results),
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'use_previous_context': self.config.use_previous_context,
                    'max_context_length': self.config.max_context_length,
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p
                }
            },
            'predictions': results
        }
        
        # JSON 저장
        output_path = os.path.join(self.config.output_dir, self.config.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # CSV 저장 (선택적)
        df_data = []
        for r in results:
            df_data.append({
                'sequence': r.get('sequence_number'),
                'image_name': r.get('image_name'),
                'prediction': r.get('prediction', ''),
                'generation_time': r.get('generation_time', 0),
                'used_context': r.get('used_context', False),
                'error': r.get('error', '')
            })
        
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(self.config.output_dir, 'sequential_predictions.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 통계 출력
        self.print_statistics(results)
        
        self.logger.info(f"결과 저장 완료: {output_path}")
    
    def print_statistics(self, results: List[Dict[str, Any]]) -> None:
        """예측 통계 출력"""
        total = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        failed = total - successful
        
        generation_times = [r.get('generation_time', 0) for r in results if 'generation_time' in r]
        avg_time = np.mean(generation_times) if generation_times else 0
        
        context_used = sum(1 for r in results if r.get('used_context', False))
        
        self.logger.info("\n" + "="*50)
        self.logger.info("예측 통계")
        self.logger.info("="*50)
        self.logger.info(f"총 이미지 수: {total}")
        self.logger.info(f"성공: {successful}")
        self.logger.info(f"실패: {failed}")
        self.logger.info(f"평균 생성 시간: {avg_time:.2f}초")
        self.logger.info(f"컨텍스트 사용: {context_used}/{total}")
        self.logger.info("="*50 + "\n")


class VLMEvaluator:
    """Vision Language Model 평가를 위한 메인 클래스"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.setup_logging()
        self.model = None
        self.processor = None
        self.rouge = Rouge()
        self.predictions = []
        self.ground_truths = []
        
    def setup_logging(self) -> None:
        """로깅 시스템 설정"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_dir}/evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def clear_memory(self) -> None:
        """GPU 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_model_and_processor(self) -> None:
        """파인튜닝된 모델 및 프로세서 로드"""
        try:
            self.logger.info(f"모델 로드 시작: {self.config.model_path}")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels
            )
            
            self.model.eval()
            self.logger.info("모델 및 프로세서 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def load_test_data(self) -> List[Dict]:
        """테스트 데이터 로드"""
        try:
            if not Path(self.config.test_data_path).exists():
                raise FileNotFoundError(f"테스트 데이터 파일을 찾을 수 없습니다: {self.config.test_data_path}")
            
            dataset = load_dataset('json', data_files=self.config.test_data_path)
            test_data = dataset['train']
            
            if self.config.max_samples:
                test_data = test_data.select(range(min(self.config.max_samples, len(test_data))))
            
            self.logger.info(f"테스트 데이터 로드 완료: {len(test_data)}개 샘플")
            return test_data
            
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {e}")
            raise
    
    def compute_text_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """텍스트 기반 평가 지표 계산"""
        metrics = {}
        
        # BLEU Score
        bleu_scores = []
        smoothing = SmoothingFunction().method4
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        metrics['bleu'] = np.mean(bleu_scores)
        
        # ROUGE Scores
        try:
            rouge_scores = self.rouge.get_scores(predictions, references, avg=True)
            metrics['rouge-1'] = rouge_scores['rouge-1']['f']
            metrics['rouge-2'] = rouge_scores['rouge-2']['f']
            metrics['rouge-l'] = rouge_scores['rouge-l']['f']
        except Exception as e:
            self.logger.warning(f"ROUGE 계산 중 오류: {e}")
            metrics['rouge-1'] = metrics['rouge-2'] = metrics['rouge-l'] = 0.0
        
        return metrics


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM 모델 평가 및 예측')
    parser.add_argument('--mode', type=str, default='predict_folder', 
                        choices=['evaluate', 'predict_folder'],
                        help='실행 모드 선택')
    parser.add_argument('--model_path', type=str, 
                        default='/workspace/Toonspace_VLM/ex_models/with_previous_toptoon_data_grok/checkpoint-10000',
                        help='모델 경로')
    parser.add_argument('--base_model_id', type=str,
                        default='huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated',
                        help='원본 모델 ID (프로세서 로드용)')
    parser.add_argument('--image_folder', type=str,
                        default='/workspace/Toonspace_VLM/data/test_image/escape_home/01',
                        help='예측할 이미지 폴더 경로')
    parser.add_argument('--use_context', type=bool, default=True,
                        help='이전 컨텍스트 사용 여부')
    parser.add_argument('--output_dir', type=str, default='/workspace/Toonspace_VLM/test/output_json',
                        help='결과 저장 경로')
    
    args = parser.parse_args()
    
    if args.mode == 'predict_folder':
        # 웹툰 폴더 예측 모드
        config = WebtoonPredictConfig(
            model_path=args.model_path,
            base_model_id=args.base_model_id,
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            use_previous_context=args.use_context
        )
        predictor = WebtoonSequentialPredictor(config)
        results = predictor.predict_folder()
        
        print(f"\n✅ 예측 완료!")
        print(f"📁 결과 파일: {args.output_dir}/sequential_predictions.json")
        print(f"📊 총 {len(results)}개 이미지 처리 완료")
        
    elif args.mode == 'evaluate':
        # 평가 모드
        config = TestConfig(
            model_path=args.model_path
        )
        evaluator = VLMEvaluator(config)
        # 평가 로직은 기존 코드 참조


if __name__ == "__main__":
    main()