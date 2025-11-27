import json
import random

BBOX_OCR_QUERIES = [
    # 빨간 박스 + 텍스트별 BBOX 요청 (가장 명확한 요청)
    "이 그림에서 붉은색 박스 안에 있는 모든 텍스트를 개별적으로 인식하고, 각 텍스트의 백분율 바운딩 박스를 추출해 줘.",
    "이미지 내 빨간 상자로 둘러싸인 텍스트들 각각의 내용과, 그 개별 백분위 BBOX 좌표를 알려줘.",
    "빨간색으로 표시된 영역 내 문자를 파악하고, 각 문자의 위치를 백분위 바운딩 박스로 추출해.",
    "이 사진에서 빨간색 테두리 안의 글씨를 모두 인식하고, 그 개별 텍스트별 BBOX를 백분율 좌표계로 알려줘.",
    "적색 상자에 있는 모든 문자를 읽어내고, 그 각각의 바운딩 박스를 백분위로 제공해.",
    
    # 간결하면서도 핵심을 포함
    "빨간 박스 내의 텍스트와 해당 텍스트의 백분율 BBOX를 모두 쌍으로 묶어 추출해 줘.",
    "붉은색 박스 영역의 OCR 텍스트와, 그 텍스트별 0~100% 좌표를 정리해.",
    "빨간색으로 마크된 부분의 텍스트를 모두 읽어내고, 각 텍스트에 대한 백분율 BBOX를 반환해.",
    "적색 구역 내의 모든 텍스트와 개별 백분위 경계 상자 데이터를 구조화하여 출력해.",
    "빨간 네모 속 텍스트와 그 텍스트별 위치를 퍼센트 BBOX로 계산해 줘.",
    
    # 다양한 표현
    "빨간색으로 하이라이트된 영역의 텍스트를 알려주고, 텍스트별 백분위 좌표의 바운딩 박스를 반환해.",
    "이 사진의 빨간색 경계 상자 속 텍스트를 인식하고, 텍스트별 bbox를 백분율 좌표계로 지정해 줘.",
    "적색 박스로 구획된 문자를 알려주고, 텍스트별 백분율 기반 경계 상자를 추출해 주세요.",
]
# --- 2. "일반 OCR" (원본 이미지)용 쿼리 ---
OCR_QUERIES = [
    "이 그림에서 문자를 찾아내고, 상대적인 백분율 위치 좌표를 돌려줘.",
    "이미지 내의 텍스트 요소를 인식하고, 그 위치를 퍼센트 기준으로 알려줘.",
    "그림 속 글씨를 추출하고, 백분율 좌표 형태로 결과값을 줘.",
    "이 사진에서 텍스트를 인지하고, 백분위 좌표를 출력해.",
    "이미지에서 모든 글자를 식별하고, 위치를 백분율 스케일로 반환해.",
    "그림의 문자 정보를 OCR로 추출하고, 백분위 좌표로 표시해 줘.",
    "이 이미지 파일에서 텍스트를 검출하고, 백분율 기반의 위치 정보를 알려줘.",
    "표시된 이미지에서 텍스트를 읽고, 위치를 퍼센티지 값으로 표시해.",
    "그림에 있는 텍스트를 파싱하고, 백분위 위치 값을 알려줘.",
    "이 이미지의 모든 텍스트 내용을 추출하고, 좌표를 0에서 100 사이의 값으로 줘.",
    "이미지에서 텍스트 블록을 찾고, 그 위치를 백분율 형식으로 알려줘.",
    "사진 속의 글자들을 인식하고, 상대적 위치를 퍼센티지로 계산하여 반환해.",
    "이 그림 자료에서 텍스트를 구분하고, 백분위 좌표계로 위치를 지정해 줘.",
    "이미지의 텍스트 성분을 추출하고, 백분율 단위로 위치를 표시해.",
    "OCR을 이용해 이 이미지의 텍스트와 백분율 위치 정보를 요청해.",
    "그림 속의 문자열을 감지하고, 해당 위치를 백분위로 변환하여 줘.",
    "이 이미지에서 텍스트를 찾고, 좌표계를 정규화(0~100%)하여 반환해.",
    "사진에서 텍스트를 찾아내고, 백분율 기반의 경계 상자 위치를 알려줘.",
    "이미지 내 글씨를 인식하고, 위치 정보를 백분위 좌표로 알려주세요.",
    "이 그림에서 문자를 읽어내고, 위치를 퍼센트 값으로 반환해.",
]


def transform_to_qa_format(input_file_path, output_file_path):
    """
    OCR 주석 딕셔너리를 "Query-Answer-TextParts" 형식의 리스트로 변환합니다.

    - 입력 형식: { "image_path": [ ... annotations ... ] }
    - 출력 형식: [ { "image_path": "...", "query": "...", "answer": "...", "text_parts": ["text1", "text2"] }, ... ]
    - 경로에 "Target_paired_visual_prompting"이 있으면 BBOX_OCR_QUERIES 사용
    - 경로에 "Target_paired"만 있으면 OCR_QUERIES 사용
    """
    
    # 1. 원본 JSON 파일 읽기
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file_path}'를 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{input_file_path}' 파일의 JSON 형식이 올바르지 않습니다.")
        return

    output_data_list = []

    # 2. 입력 데이터 순회 (key: image_path, value: annotations_list)
    skip_number = 0 
    for image_path, annotations_list in input_data.items():
        
        current_query_list = None
        
        # 3. 경로명에 따른 쿼리 리스트 선택
        if "Target_paired_visual_prompting" in image_path:
            current_query_list = BBOX_OCR_QUERIES
        elif "Target_paired" in image_path:
            current_query_list = OCR_QUERIES
        
        if not current_query_list:
            print(f"경고: 쿼리 조건을 찾을 수 없습니다. 경로를 건너뜁니다: '{image_path}'")
            continue
            
        # 4. 각 이미지의 annotation 리스트 순회 (BBOX 마다 Q/A 1개 생성)
        
        # 7. "query" 랜덤 선택 (선택된 리스트에서)
        # (참고: 원본 코드는 이미지당 3개의 쿼리를 *샘플링*합니다.)
        query_string_list = random.sample(current_query_list, min(len(current_query_list), 3)) # 3개 또는 그 이하
        
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


# --- 코드 실행 예제 ---

# 2. 변환 함수 실행
output_filename = "ocr_description/total(bbox_normal)_ocr_dataset_2F.json"
input_file_name = '/workspace/Toonspace_VLM/data/OCR(bbox_merge)_json_data/total_korean_ocr.json'
transform_to_qa_format(input_file_name, output_filename)


# 3. 결과 확인 (생성된 파일 내용 출력)
# print(f"\n--- 생성된 '{output_filename}' 파일 내용 ---")
# try:
#     with open(output_filename, 'r', encoding='utf-8') as f:
#         print(f.read())
# except FileNotFoundError:
#     print("출력 파일을 생성하지 못했습니다.")