"""
Vision Language Model 예측 및 평가 통합 스크립트 (웹툰 순차 추론 및 OCR 평가 지원)

이 모듈은 파인튜닝된 Qwen2.5-VL 모델의 예측 및 평가를 단일 클래스 내에서 구현합니다.
'predict' 모드에서는 웹툰 이미지를 순차적으로 분석하며 컨텍스트를 누적 처리하고,
'evaluate' 모드에서는 테스트 데이터셋을 사용하여 텍스트 생성 및 OCR/Detection 성능을 평가합니다.

작성자: Assistant
Year: 2025
Month: 9
Day: 17
(Refactored on: 2025-10-16)
"""
import gc
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from nltk import download, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from qwen_vl_utils import process_vision_info
# 평가 지표 관련 라이브러리
from rouge import Rouge
from bert_score import score as bert_score
from tqdm import tqdm
from transformers import (AutoProcessor,
                          Qwen2_5_VLForConditionalGeneration)

# NLTK 데이터 다운로드 (최초 실행 시)
try:
    download('punkt', quiet=True)
    download('wordnet', quiet=True)
except Exception:
    pass

@dataclass
class VLMConfig:
    """VLM 예측 및 평가 설정을 통합 관리하는 데이터클래스"""
    
    # 📁 데이터 및 모델 경로
    model_path: str = ''
    base_model_id: str = "Qwen/Qwen2-VL-7B-Instruct" 

    # ➡️ 예측 모드 설정
    image_folder: str = '/workspace/Toonspace_VLM/webtoon_images'
    use_previous_context: bool = False
    max_context_length: int = 2000
    context_summary_length: int = 500
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.webp'])
    # base_prompt: str = "이 웹툰 이미지를 분석하여 대사, 효과음, 캐릭터 행동, 감정을 JSON 형식으로 추출해주세요."
#     context_prompt_template: str = """이전 장면 요약:
# {previous_context}

# 현재 이미지를 위의 컨텍스트를 고려하여 분석해주세요. 이 웹툰 이미지의 대사, 효과음, 캐릭터 행동, 감정을 JSON 형식으로 추출해주세요."""

    # 📊 평가 모드 설정
    test_data_path: str = '/workspace/Toonspace_VLM/test/OCR_test_dataset.json'
    compute_bertscore: bool = True
    compute_json_metrics: bool = True
    
    ##### NEW/MODIFIED START #####
    compute_ocr_metrics: bool = True # OCR/Detection 메트릭 계산 여부
    iou_threshold: float = 0.5       # IoU 임계값
    ##### NEW/MODIFIED END #####

    # ⚙️ 모델 및 생성 공통 설정
    output_dir: str = 'results'
    predictions_file: str = 'predictions.json'
    metrics_file: str = 'evaluation_metrics.json'
    batch_size: int = 1
    max_samples: Optional[int] = None
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 960 * 28 * 28

    # 📝 시스템 메시지
    system_message: str = field(default="""
    당신은 웹툰 이미지 분석 전문가입니다. 웹툰의 연속된 장면을 분석하여 스토리의 흐름을 이해하고,
    각 장면의 대사, 효과음, 캐릭터의 행동과 감정, 서사적 맥락을 정확히 추출합니다.
    이전 장면의 맥락을 고려하여 현재 장면을 더 정확하게 해석하고,
    모든 결과를 구조화된 JSON 형식으로 제공합니다.
    """)

class WebtoonVLM:
    """웹툰 VLM 예측 및 평가를 위한 통합 클래스"""

    def __init__(self, config: VLMConfig):
        self.config = config
        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.rouge = Rouge()
        
        self.load_model_and_processor()

    def setup_logging(self) -> None:
        """로깅 시스템 설정"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / 'vlm_runner.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_model_and_processor(self) -> None:
        """파인튜닝된 모델 및 프로세서 로드"""
        if self.model is not None:
            self.logger.info("모델이 이미 로드되었습니다.")
            return
        try:
            self.logger.info(f"모델 로드 시작: {self.config.model_path}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.config.base_model_id, min_pixels=self.config.min_pixels, max_pixels=self.config.max_pixels, trust_remote_code=True
            )
            self.model.eval()
            self.logger.info("모델 및 프로세서 로드 완료")
        except Exception as e:
            self.logger.error(f"모델 로드 중 치명적 오류 발생: {e}")
            raise

    def clear_memory(self) -> None:
        """GPU 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def run(self, mode: str):
        """메인 실행 함수: 모드에 따라 예측 또는 평가 실행"""
        if mode == 'predict':
            return self._run_prediction()
        elif mode == 'evaluate':
            return self._run_evaluation()
        else:
            self.logger.error(f"지원하지 않는 모드입니다: {mode}. 'predict' 또는 'evaluate'를 선택해주세요.")
            raise ValueError(f"Invalid mode: {mode}")
    
    # ... (이전 코드의 예측 모드 관련 메서드들은 변경되지 않았으므로 생략) ...
    def _run_prediction(self) -> List[Dict[str, Any]]:
        folder_path = self.config.image_folder
        self.logger.info(f"폴더 예측 모드 시작: {folder_path}")
        image_files = self._get_sorted_image_files(folder_path)
        if not image_files: return []
        all_results, context_history = [], []
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            prompt = self._build_context_prompt(context_history)
            result = self._predict_single(image_path=image_path, query=prompt)
            result['sequence_number'] = idx + 1
            result['used_context'] = self.config.use_previous_context and bool(context_history)
            if self.config.use_previous_context:
                context_summary = self._extract_key_info_from_prediction(result['prediction'])
                context_history.append(context_summary)
                if len(context_history) > 5: context_history.pop(0)
            all_results.append(result)
            if (idx + 1) % 10 == 0: self.clear_memory()
        self._save_prediction_results(all_results, folder_path)
        self.logger.info(f"예측 완료: 총 {len(all_results)}개 이미지 처리")
        return all_results
    def _get_sorted_image_files(self, folder_path: str) -> List[Path]:
        folder = Path(folder_path)
        if not folder.is_dir(): raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
        image_files = [p for p in folder.iterdir() if p.suffix.lower() in self.config.image_extensions]
        def natural_sort_key(path): return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.name)]
        image_files.sort(key=natural_sort_key)
        self.logger.info(f"총 {len(image_files)}개의 이미지 파일 발견")
        return image_files
    def _build_context_prompt(self, context_history: List[str]) -> str:
        if not self.config.use_previous_context or not context_history: return self.config.base_prompt
        context_summary = "\n".join([f"장면 {i+1}: {ctx}" for i, ctx in enumerate(context_history)])
        if len(self.processor.tokenizer.encode(context_summary)) > self.config.max_context_length:
            context_summary = context_summary[-self.config.max_context_length:]
        return self.config.context_prompt_template.format(previous_context=context_summary)
    def _extract_key_info_from_prediction(self, prediction: str) -> str:
        try:
            pred_json = json.loads(prediction)
            summary_parts = []
            if 'dialogues' in pred_json and pred_json['dialogues']: summary_parts.append(f"대사: {', '.join([d.get('text', '') for d in pred_json['dialogues'][:2]])}")
            if 'characters' in pred_json and pred_json['characters']: summary_parts.append(f"행동: {', '.join([c.get('action', '') for c in pred_json['characters'][:2] if c.get('action')])}")
            return ' | '.join(summary_parts) if summary_parts else prediction[:self.config.context_summary_length]
        except (json.JSONDecodeError, TypeError): return prediction[:self.config.context_summary_length]
    def _save_prediction_results(self, results: List[Dict], source_folder: str):
        output_data = {'metadata': {'source_folder': source_folder, 'model_path': self.config.model_path, 'total_images': len(results), 'timestamp': datetime.now().isoformat(), 'config': self.config.__dict__}, 'predictions': results}
        output_path = Path(self.config.output_dir) / self.config.predictions_file
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(output_data, f, ensure_ascii=False, indent=2)
        pd.DataFrame(results).to_csv(Path(self.config.output_dir) / 'predictions.csv', index=False, encoding='utf-8-sig')
        self.logger.info(f"예측 결과 저장 완료: {output_path}")

    # ===================================================================
    # 평가 모드 (Evaluation Mode) - 수정됨
    # evaluate - query : OCR 만 사용. 
    # ===================================================================

    def _run_evaluation(self) -> Dict[str, Any]:
        """모델 예측 및 평가 지표 계산"""
        self.logger.info("평가 모드 시작...")
        test_data = self._load_test_data()
        results_with_metadata = []
        
        start_time = time.time()
        for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                query = sample['query']
                result = self._predict_single(image_path=sample['image_path'], query=query)
                result['ground_truth'] = sample['answer']
                results_with_metadata.append(result)
            except Exception as e:
                self.logger.error(f"샘플 {idx+1} ({sample.get('image_path')}) 처리 중 오류: {e}")
                results_with_metadata.append({"image_path": sample.get('image_path'), "prediction": "[ERROR]", "ground_truth": sample.get('answer'), "error": str(e)})
            if (idx + 1) % 10 == 0: self.clear_memory()

        total_time = time.time() - start_time
        self.logger.info(f"평가 완료. 총 소요 시간: {total_time:.2f}초")

        predictions = [r['prediction'] for r in results_with_metadata]
        ground_truths = [r['ground_truth'] for r in results_with_metadata]
        
        metrics = self._calculate_all_metrics(predictions, ground_truths)
        self._save_evaluation_results(metrics, results_with_metadata)
        
        return metrics

    def _load_test_data(self) -> List[Dict]:
        """테스트 데이터셋 로드"""
        try:
            dataset = load_dataset('json', data_files=self.config.test_data_path)['train']
            # dataset = dataset.select(range(20))
            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
            self.logger.info(f"테스트 데이터 로드 완료: {len(dataset)}개 샘플")
            return dataset
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류: {e}")
            raise

    ##### NEW/MODIFIED START #####
    def _parse_ocr_string(self, ocr_str: str) -> List[Dict[str, Union[str, List[float]]]]:
        """ 'TEXT : [x1, y1, x2, y2]' 형식의 문자열을 파싱합니다. """
        entries = []
        if not isinstance(ocr_str, str):
            return []
        for line in ocr_str.strip().split('\n'):
            try:
                parts = line.split(' : ')
                if len(parts) == 2:
                    text = parts[0]
                    # 대괄호 안의 내용만 추출하여 파싱
                    bbox_str = parts[1].strip()
                    bbox = json.loads(bbox_str)
                    if len(bbox) == 4:
                        entries.append({'text': text, 'box': [float(coord) for coord in bbox]})
            except (json.JSONDecodeError, IndexError, ValueError) as e:
                self.logger.debug(f"OCR 문자열 파싱 실패: '{line}'. 오류: {e}")
                continue
        return entries

    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        """두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        unionArea = float(boxAArea + boxBArea - interArea)
        return interArea / unionArea if unionArea > 0 else 0.0
    def _compute_map(self, predictions: List[str], ground_truths: List[str], iou_threshold: float) -> float:
        """
        데이터셋 전체에 대한 Average Precision (AP)을 계산합니다.
        참고: 신뢰도 점수가 없으므로 모든 예측의 신뢰도를 1.0으로 가정합니다.
        """
        all_preds = []
        total_gt_boxes = 0

        # 1. 모든 예측과 정답을 하나의 리스트로 통합
        for i, (pred_str, gt_str) in enumerate(zip(predictions, ground_truths)):
            pred_boxes = self._parse_ocr_string(pred_str)
            gt_boxes = self._parse_ocr_string(gt_str)
            total_gt_boxes += len(gt_boxes)
            
            # 각 예측에 이미지 인덱스와 신뢰도(가짜) 추가
            for pred_box in pred_boxes:
                all_preds.append({'image_id': i, 'confidence': 1.0, 'box': pred_box})
        
        # 신뢰도 점수 기준 내림차순 정렬 (현재는 모두 1.0이라 의미는 없지만, 알고리즘상 필요)
        all_preds.sort(key=lambda x: x['confidence'], reverse=True)

        # 2. 각 예측에 대해 TP/FP 판별
        gt_matched_map = {i: [False] * len(self._parse_ocr_string(gt_str)) for i, gt_str in enumerate(ground_truths)}
        
        tp_fp_list = []
        for pred in all_preds:
            image_id = pred['image_id']
            gt_boxes = self._parse_ocr_string(ground_truths[image_id])
            
            best_iou = 0
            best_gt_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if not gt_matched_map[image_id][i]:
                    iou = self._calculate_iou(pred['box']['box'], gt_box['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            if best_gt_idx != -1 and best_iou >= iou_threshold and \
               pred['box']['text'].strip() == gt_boxes[best_gt_idx]['text'].strip():
                gt_matched_map[image_id][best_gt_idx] = True
                tp_fp_list.append(1)  # True Positive
            else:
                tp_fp_list.append(0)  # False Positive

        # 3. Precision-Recall 곡선 계산
        if total_gt_boxes == 0:
            return 0.0
            
        tp_cumsum = np.cumsum(tp_fp_list)
        precision_curve = [tp_cumsum[i] / (i + 1) for i in range(len(tp_fp_list))]
        recall_curve = [tp_cumsum[i] / total_gt_boxes for i in range(len(tp_fp_list))]

        # 4. AP (Area under curve) 계산
        precision_curve = np.array([0.0] + precision_curve)
        recall_curve = np.array([0.0] + recall_curve)
        
        # 우하향 곡선으로 보정
        for i in range(len(precision_curve) - 2, -1, -1):
            precision_curve[i] = max(precision_curve[i], precision_curve[i+1])

        # recall 값이 변하는 지점만 선택하여 면적 계산
        recall_change_indices = np.where(recall_curve[1:] != recall_curve[:-1])[0]
        ap = np.sum((recall_curve[recall_change_indices + 1] - recall_curve[recall_change_indices]) * precision_curve[recall_change_indices + 1])
        
        return ap
    def _compute_detection_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """OCR 및 BBox 결과를 바탕으로 Precision, Recall, F1-score, Avg IoU, mAP를 계산합니다."""
        total_tp, total_fp, total_fn = 0, 0, 0
        iou_scores_for_tps = [] # TP들의 IoU 점수를 저장할 리스트

        for pred_str, gt_str in zip(predictions, ground_truths):
            pred_boxes = self._parse_ocr_string(pred_str)
            gt_boxes = self._parse_ocr_string(gt_str)

            if not gt_boxes and not pred_boxes:
                continue

            tp = 0
            gt_matched = [False] * len(gt_boxes)

            # 각 예측 박스를 최적의 정답 박스와 매칭
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for i, gt_box in enumerate(gt_boxes):
                    if gt_matched[i]: continue
                    iou = self._calculate_iou(pred_box['box'], gt_box['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                # TP 조건: IoU 임계값 초과 및 텍스트 일치
                if best_gt_idx != -1 and best_iou >= self.config.iou_threshold and \
                   pred_box['text'].strip() == gt_boxes[best_gt_idx]['text'].strip():
                    tp += 1
                    gt_matched[best_gt_idx] = True
                    iou_scores_for_tps.append(best_iou) # TP일 때의 IoU 점수 저장
            
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - sum(gt_matched)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # --- 기본 메트릭 계산 ---
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # --- 평균 IoU 계산 ---
        average_iou = np.mean(iou_scores_for_tps) if iou_scores_for_tps else 0.0

        # --- mAP 계산 ---
        # 클래스가 하나이므로 mAP@0.5는 AP@0.5와 동일합니다.
        map_score = self._compute_map(predictions, ground_truths, self.config.iou_threshold)

        return {
            'ocr_precision': precision,
            'ocr_recall': recall,
            'ocr_f1_score': f1_score,
            'average_iou': average_iou,
            f'mAP_at_{self.config.iou_threshold}': map_score
        }

    def _calculate_all_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """모든 평가 지표를 계산하고 통합"""
        metrics = {}
        valid_pairs = [(p, r) for p, r in zip(predictions, ground_truths) if p and r and p != "[ERROR]"]
        if not valid_pairs:
            self.logger.warning("계산할 유효한 예측-정답 쌍이 없습니다.")
            return {"error": "No valid prediction-reference pairs found."}
            
        filtered_preds, filtered_refs = zip(*valid_pairs)
        
        # Text Metrics (기존)
        try:
            rouge_scores = self.rouge.get_scores(filtered_preds, filtered_refs, avg=True)
            metrics['rouge-l'] = rouge_scores['rouge-l']['f']
        except Exception as e: self.logger.warning(f"ROUGE 계산 중 오류: {e}")
        
        # BERTScore (기존)
        if self.config.compute_bertscore:
            try:
                _, _, F1 = bert_score(list(filtered_preds), list(filtered_refs), lang="ko", device=self.device)
                metrics['bert_score_f1'] = F1.mean().item()
            except Exception as e: self.logger.warning(f"BERTScore 계산 중 오류: {e}")

        # JSON Metrics (기존)
        if self.config.compute_json_metrics:
            # 이 부분은 현재 OCR 평가와는 무관하므로, 필요 시 JSON 형식에 맞게 수정 필요
            pass

        # OCR/Detection Metrics (신규)
        if self.config.compute_ocr_metrics:
            self.logger.info(f"OCR/Detection 메트릭 계산 (IoU Threshold: {self.config.iou_threshold})...")
            detection_metrics = self._compute_detection_metrics(predictions, ground_truths)
            metrics.update(detection_metrics)

        return metrics
    ##### NEW/MODIFIED END #####

    def _save_evaluation_results(self, metrics: Dict, results_with_metadata: List[Dict]):
        """평가 결과 및 메트릭 저장"""
        metadata = {"timestamp": datetime.now().isoformat(), "model_path": self.config.model_path, "test_data_path": self.config.test_data_path}
        
        metrics_output = {"metadata": metadata, "metrics": metrics}
        metrics_path = Path(self.config.output_dir) / self.config.metrics_file
        with open(metrics_path, 'w', encoding='utf-8') as f: json.dump(metrics_output, f, ensure_ascii=False, indent=4)
        self.logger.info(f"평가 메트릭 저장 완료: {metrics_path}")

        predictions_output = {"metadata": metadata, "predictions": results_with_metadata}
        predictions_path = Path(self.config.output_dir) / self.config.predictions_file
        with open(predictions_path, 'w', encoding='utf-8') as f: json.dump(predictions_output, f, ensure_ascii=False, indent=2)
        self.logger.info(f"개별 예측 결과 저장 완료: {predictions_path}")

    @torch.no_grad()
    def _predict_single(self, image_path: Union[str, Path], query: str) -> Dict[str, Any]:
        """단일 이미지에 대한 예측을 수행하는 공통 함수"""
        try:
            image = Image.open(image_path).convert('RGB')
            messages = [{"role": "system", "content": self.config.system_message}, {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)
            start_time = time.time()
            gen_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, temperature=self.config.temperature, top_p=self.config.top_p, do_sample=self.config.do_sample)
            gen_time = time.time() - start_time
            gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
            prediction = self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return {'image_path': str(image_path), 'image_name': Path(image_path).name, 'query': query, 'prediction': prediction, 'generation_time': gen_time}
        except Exception as e:
            self.logger.error(f"이미지 {image_path} 예측 중 오류 발생: {e}")
            return {'image_path': str(image_path), 'image_name': Path(image_path).name, 'query': query, 'prediction': "[PREDICTION_ERROR]", 'error': str(e)}

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM 모델 예측 및 평가 통합 스크립트')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['predict', 'evaluate'])
    parser.add_argument('--model_path', type=str, default='/workspace/Toonspace_VLM/ex_models/at_once_ocr_description')
    parser.add_argument('--base_model_id', type=str, default='huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated')
    parser.add_argument('--image_folder', type=str, default='/workspace/Toonspace_VLM/data/test_image/escape_home/01')
    parser.add_argument('--test_data_path', type=str, default='/workspace/Toonspace_VLM/test/OCR_test_dataset.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--use_context', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    
    config = VLMConfig(
        model_path=args.model_path,
        base_model_id=args.base_model_id,
        image_folder=args.image_folder,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        use_previous_context=args.use_context
    )
    
    runner = WebtoonVLM(config)
    
    if args.mode == 'predict':
        results = runner.run(mode='predict')
        print(f"\n✅ 예측 완료!")
        print(f"📁 결과 파일: {Path(args.output_dir) / config.predictions_file}")
        print(f"📊 총 {len(results)}개 이미지 처리 완료")
        
    elif args.mode == 'evaluate':
        metrics = runner.run(mode='evaluate')
        print("\n✅ 평가 완료!")
        print("📊 주요 평가 지표:")
        if metrics:
            for metric, value in metrics.items():
                print(f"- {metric}: {value:.4f}")
        print(f"📁 상세 결과 파일: {Path(args.output_dir) / config.metrics_file}")

if __name__ == "__main__":
    main()