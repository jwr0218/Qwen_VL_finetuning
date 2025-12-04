import json
import random
from dataclasses import dataclass, field
from typing import List
from query_class import OCRPrompts


Prompt = OCRPrompts()

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
        print(query_string_list)
        exit()
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
input_file_name = '/workspace/Toonspace_VLM/data/test2.json'
transform_to_qa_format(input_file_name, output_filename)


# 3. 결과 확인 (생성된 파일 내용 출력)
# print(f"\n--- 생성된 '{output_filename}' 파일 내용 ---")
# try:
#     with open(output_filename, 'r', encoding='utf-8') as f:
#         print(f.read())
# except FileNotFoundError:
#     print("출력 파일을 생성하지 못했습니다.")