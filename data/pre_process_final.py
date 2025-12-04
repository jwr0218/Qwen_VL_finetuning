import json
import os
from dataclasses import dataclass, field
from typing import List
from query_class import OCRPrompts



def transform_to_qa_format(input_source, output_file_path):
    # 1. 입력 소스 확인 및 데이터 로드
    if isinstance(input_source, str):
        # 입력이 문자열이면 파일 경로로 간주
        try:
            with open(input_source, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"성공: '{input_source}' 파일을 읽었습니다.")
        except FileNotFoundError:
            print(f"오류: 입력 파일 '{input_source}'를 찾을 수 없습니다.")
            return
        except json.JSONDecodeError:
            print(f"오류: '{input_source}' 파일의 JSON 형식이 올바르지 않습니다.")
            return
            
    elif isinstance(input_source, (dict, list)):
        # 입력이 이미 JSON 객체(딕셔너리 또는 리스트)인 경우
        input_data = input_source
        
    else:
        print("오류: 입력값은 파일 경로(str) 또는 JSON 데이터(dict, list)여야 합니다.")
        return

    output_data_list = []

    # 2. 입력 데이터 순회 (key: image_path, value: annotations_list)
    skip_number = 0 
    for image_path, annotations_list in input_data.items():
        
        current_query_list = None
            
        # 4. 각 이미지의 annotation 리스트 순회 (BBOX 마다 Q/A 1개 생성)
        
        # 7. "query" 랜덤 선택 (선택된 리스트에서)
        # (참고: 원본 코드는 이미지당 3개의 쿼리를 *샘플링*합니다.)
        query_string_list = Prompt.get_random_ocr_query(10)
        # print(query_string_list)
        # exit()
        for query_string in query_string_list:
            
            scene_answer_part = []
            scene_text_parts = []  # (NEW) 텍스트만 저장할 리스트 초기화
            
            for annotation in annotations_list:
                try:
                    # 5. 텍스트 및 바운딩 박스 좌표 추출
                    text = annotation.get('text') # (수정됨) 텍스트를 변수로 먼저 추출
                    
                    if text is None: 
                        skip_number +=1 
                        continue
                        
                    # (NEW) 가중치를 줄 텍스트를 리스트에 추가
                    scene_text_parts.append(text)

                    bbox = annotation.get("bbox_pixel", {})
                    x1 = bbox['x1']
                    y1 = bbox['y1']
                    x2 = bbox['x2']
                    y2 = bbox['y2']

                    # 6. "answer" 문자열 포맷팅
                    answer_string = f"{text} : [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]"
                    
                    scene_answer_part.append(answer_string)
                    
                except KeyError as e:
                    print(f"경고: 주석에서 필수 키({e})를 찾을 수 없습니다. (이미지: {image_path})")
                except Exception as e:
                    print(f"오류: 주석 처리 중 에러 발생: {e} (이미지: {image_path})")
            
            # (수정됨) 텍스트 부분이 하나도 없으면 이 Q/A 쌍은 생성하지 않음
            if not scene_answer_part:
                continue

            # 8. (수정됨) "text_parts"를 포함하여 Q/A 딕셔너리 생성
            scene_answer_all = "\n".join(scene_answer_part)
            
            qa_pair = {
                "image_path": image_path,
                "query": query_string,
                "answer": scene_answer_all,
                "text_parts": scene_text_parts  # (NEW) 수집된 텍스트 리스트 추가
            }
            output_data_list.append(qa_pair)
            
    # 9. 변환된 데이터를 새 JSON 파일로 저장
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data_list, f, ensure_ascii=False, indent=2)
        
        print(f"\n변환 완료! 총 {len(output_data_list)}개의 Q/A 쌍이 '{output_file_path}'에 저장되었습니다.")
        
    except IOError as e:
        print(f"오류: '{output_file_path}' 파일 쓰기에 실패했습니다. {e}")

    print(f'경고 : None 값입니다. 해당 annotation을 {skip_number}만큼 skip합니다.')





def preprocess_ocr_data(input_file_path, output_file_path='',SAVE_PROCESS = True):
    """
    OCR annotation JSON 파일을 VLM 학습용 데이터셋(딕셔너리 형태)으로 전처리합니다.

    - 원본 형식: Label Studio와 유사한 복잡한 JSON 리스트
    - 출력 형식: { "image_path": [ ... annotations ... ] } 딕셔너리
    
    주요 변환 작업:
    1. image_path를 딕셔너리의 'key'로 사용합니다.
    2. 'id'가 동일한 텍스트('transcription')와 라벨('cls_label') 주석을 병합합니다.
    3. (x, y, width, height) 좌표를 (x1, y1, x2, y2)로 변환합니다.
    4. 텍스트가 없는 라벨(e.g., bb_text, bx_text)은 'text' 필드를 null(None)로 설정합니다.
    """
    
    # 1. 원본 JSON 파일 읽기
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file_path}'를 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{input_file_path}' 파일의 JSON 형식이 올바르지 않습니다.")
        return

    # 2. 최종 출력을 딕셔너리 형태로 초기화
    preprocessed_data = {}

    # 3. 각 이미지(task) 순회
    for task in original_data:


        # 'id'를 기준으로 텍스트와 라벨을 병합하기 위한 임시 딕셔너리
        merged_annotations = {}

        # 'annotations' 리스트 순회 (보통 1개)
        for annotation in task.get('annotations', []):
            image_path = annotation.get('image_path')
            image_path = '/DATA/toptoon/'+image_path
            if not image_path:
                print(f"경고: 'image_path'가 없는 task {task.get('id')}를 건너뜁니다.")
                continue
            
            # 'result' 리스트 순회 (실제 OCR 결과)
            for result in annotation.get('result', []):
                
                result_id = result.get('id')
                if not result_id:
                    continue # id가 없는 주석은 병합 불가

                # 'id'가 처음 등장하면 새 항목 생성
                if result_id not in merged_annotations:
                    merged_annotations[result_id] = {}
                
                value = result.get('value', {})
                
                # 4. 텍스트, 라벨, 바운딩 박스 정보 병합
                
                # 텍스트 정보 추출 ('transcription')
                if 'text' in value and value.get('text'):
                    merged_annotations[result_id]['text'] = value['text'][0]
                
                # 라벨 정보 추출 ('cls_label')
                if 'rectanglelabels' in value and value.get('rectanglelabels'):
                    merged_annotations[result_id]['labels'] = value['rectanglelabels']
                
                # 바운딩 박스 정보 추출 (텍스트, 라벨 공통)
                # 병합 과정에서 한 번만 저장되면 됨
                if 'bbox' not in merged_annotations[result_id] and 'x' in value:
                    merged_annotations[result_id]['bbox'] = {
                        "x": value['x'],
                        "y": value['y'],
                        "width": value['width'],
                        "height": value['height']
                    }

        # 5. 병합된 주석들을 최종 출력 형식으로 변환
        image_annotations_list = []
        for result_id, merged_anno in merged_annotations.items():
            
            # 바운딩 박스 정보가 없는 불완전한 주석은 건너뛰기
            bbox = merged_anno.get('bbox')
            if not bbox:
                print(f"경고: {image_path}의 주석(id: {result_id})에 bbox 정보가 없습니다.")
                continue

            # (x, y, w, h) -> (x1, y1, x2, y2) 변환
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = bbox['x'] + bbox['width']
            y2 = bbox['y'] + bbox['height']
            
            final_annotation = {
                "bbox_pixel": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                # 라벨이 없는 경우 빈 리스트, 텍스트가 없는 경우 None (json의 null)
                "labels": merged_anno.get('labels', []),
                "text": merged_anno.get('text', None) 
            }
            image_annotations_list.append(final_annotation)

        # 6. 최종 딕셔너리에 image_path를 key로 하여 주석 리스트 저장
        preprocessed_data[image_path] = image_annotations_list
    
    # 7. 전처리된 데이터를 새 JSON 파일로 저장
    if SAVE_PROCESS :

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n전처리 완료! 결과가 '{output_file_path}'에 저장되었습니다.")
            
        except IOError as e:
            print(f"오류: '{output_file_path}' 파일 쓰기에 실패했습니다. {e}")

    return preprocessed_data

    



def is_contained(inner_bbox, outer_bbox):
    """
    inner_bbox가 outer_bbox에 완전히 포함되는지 확인합니다.
    (좌표는 0-100 정규화 기준)
    """
    return (inner_bbox['x1'] >= outer_bbox['x1'] and
            inner_bbox['y1'] >= outer_bbox['y1'] and
            inner_bbox['x2'] <= outer_bbox['x2'] and
            inner_bbox['y2'] <= outer_bbox['y2'])

def merge_text_in_bubbles(input_source, output_path,SAVE_PROCESS = True):
    """
    input_source가 파일 경로(str)이면 JSON 파일을 읽고,
    JSON 객체(dict, list)이면 그대로 사용하여
    bb_text가 포함하는 텍스트들을 병합하여 새 JSON 파일을 생성합니다.
    """
    
    input_data = None

    # 1. 입력 소스 확인 및 데이터 로드
    if isinstance(input_source, str):
        # 입력이 문자열이면 파일 경로로 간주
        try:
            with open(input_source, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"성공: '{input_source}' 파일을 읽었습니다.")
        except FileNotFoundError:
            print(f"오류: 입력 파일 '{input_source}'를 찾을 수 없습니다.")
            return
        except json.JSONDecodeError:
            print(f"오류: '{input_source}' 파일의 JSON 형식이 올바르지 않습니다.")
            return
            
    elif isinstance(input_source, (dict, list)):
        # 입력이 이미 JSON 객체(딕셔너리 또는 리스트)인 경우
        input_data = input_source
        
    else:
        print("오류: 입력값은 파일 경로(str) 또는 JSON 데이터(dict, list)여야 합니다.")
        return
    preprocessed_data = {}

    # 2. 각 이미지를 순회 (key: image_path)
    for image_path, annotations_list in input_data.items():
        
        new_annotations_list = []
        
        # 3. 주석을 '컨테이너'(말풍선)와 '콘텐츠'(텍스트)로 분리
        
        # 'bb_text' 라벨을 가진 모든 박스 (컨테이너 후보)
        containers = [
            (i, anno) for i, anno in enumerate(annotations_list) 
            if "bb_text" in anno["labels"]
        ]
        
        # 'bb_text'가 아니면서 'text'가 있는 모든 박스 (콘텐츠 후보)
        contents = [
            (i, anno) for i, anno in enumerate(annotations_list) 
            if "bb_text" not in anno["labels"] and anno.get("text") is not None
        ]
        
        # 'text'가 없는 기타 박스들 (예: bx_text)
        other_boxes = [
            anno for anno in annotations_list 
            if anno.get("text") is None and "bb_text" not in anno["labels"]
        ]

        # 4. 병합 처리를 위해 소모된 객체의 인덱스를 추적
        consumed_container_indices = set()
        consumed_content_indices = set()

        # 5. 각 컨테이너(bb_text)를 순회하며 포함된 콘텐츠 찾기
        for i, container_anno in containers:
            container_bbox = container_anno["bbox_pixel"]
            contained_texts = [] # (bbox, text) 튜플 저장

            # 6. 모든 콘텐츠를 순회
            for j, content_anno in contents:
                if j in consumed_content_indices: # 이미 다른 말풍선에 포함됨
                    continue
                
                content_bbox = content_anno["bbox_pixel"]
                
                # 7. 콘텐츠가 컨테이너에 포함되는지 확인
                if is_contained(content_bbox, container_bbox):
                    contained_texts.append(
                        (content_bbox, content_anno["text"])
                    )
                    consumed_content_indices.add(j) # 이 콘텐츠는 소모됨

            # 8. 포함된 텍스트가 있다면, 병합된 새 객체 생성
            if contained_texts:
                # '순서에 맞게' -> y1 (top) 좌표 기준 오름차순 정렬
                contained_texts.sort(key=lambda item: item[0]['y1'])
                
                # 텍스트 병합 (공백으로 구분)
                merged_text = " ".join([text for bbox, text in contained_texts])
                
                # 새 병합 객체 추가
                new_annotations_list.append({
                    "bbox_pixel": container_bbox,
                    "labels": ["bb_text"], # 라벨을 'bb_text'로 고정
                    "text": merged_text
                })
                consumed_container_indices.add(i) # 이 컨테이너는 소모됨
        
        # 9. 병합되지 않은 나머지 객체들 다시 추가
        
        # 텍스트를 포함하지 않았던 'bb_text' (원본 유지)
        for i, anno in containers:
            if i not in consumed_container_indices:
                new_annotations_list.append(anno)
                
        # 어떤 말풍선에도 포함되지 않았던 'text' (원본 유지)
        for j, anno in contents:
            if j not in consumed_content_indices:
                new_annotations_list.append(anno)
                
        # 'text'가 없는 기타 박스들 (원본 유지)
        new_annotations_list.extend(other_boxes)

        # 10. 최종 결과를 딕셔너리에 저장
        preprocessed_data[image_path] = new_annotations_list

    # 11. 병합된 데이터를 새 JSON 파일로 저장
    if SAVE_PROCESS:

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
            print(f"\n병합 완료! 결과가 '{output_path}'에 저장되었습니다.")
            
        except IOError as e:
            print(f"오류: '{output_path}' 파일 쓰기에 실패했습니다. {e}")

    return preprocessed_data


Prompt = OCRPrompts()

if __name__ == '__main__':

    
    input_filename = "/workspace/generate_grok/json/raw_data/Target_paired_duplicatedv20.json"
    
    OUTPUT_JSON_PATH = "/workspace/Toonspace_VLM/data/ocr_description/for_train_dataset.json"
    
    SAVE_PROCESS = False
    INPUT_IMMEDIATE_PROCESS1 = '/workspace/Toonspace_VLM/data/intermediate_dataset/INTERMEDIATE_DATASET1.json'
    INPUT_IMMEDIATE_PROCESS2 = '/workspace/Toonspace_VLM/data/intermediate_datasetINTERMEDIATE_DATASET2.json'

    
    json_data= preprocess_ocr_data(input_filename, INPUT_IMMEDIATE_PROCESS1,SAVE_PROCESS)
    merged_data = merge_text_in_bubbles(json_data, INPUT_IMMEDIATE_PROCESS2,SAVE_PROCESS)

    transform_to_qa_format(merged_data, OUTPUT_JSON_PATH)

    
    print(f"\n--- PATH : '{OUTPUT_JSON_PATH}' \t 에 저장되었습니다.")


