import json
import os
from PIL import Image, ImageDraw

# --- ⚠️ 중요: 사용자 설정이 필요한 변수들 ---

# 1. 이전 단계에서 생성한 입력 JSON 파일 경로
INPUT_JSON_PATH = "/workspace/generate_grok/json/OCR(bbox_merge)_json_data/merged_ocr_data_korea.json"

# 2. (수정됨!) 원본 경로에서 제거할 접두사
#    이 부분이 OUTPUT_DIR 경로로 대체됩니다.
JSON_PATH_PREFIX = "/DATA/toptoon/Target_paired/"

# 3. (수정됨!) 새 이미지를 저장할 기본 폴더
#    이 경로 뒤에 "Target_paired/"가 제거된 나머지 경로가 붙습니다.
OUTPUT_DIR = "/DATA/toptoon/Target_paired_visual_prompting"

# 4. 바운딩 박스가 그려진 새 이미지가 저장될 폴더
#    (참고: 변수명이 중복되나, 기존 코드 유지를 위해 OUTPUT_DIR을 그대로 사용)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "visual_prompt_annotations.json")

# 5. 바운딩 박스 설정 (색상, 두께)
BOX_COLOR = "red"
BOX_WIDTH = 3

# ----------------------------------------------

# (수정됨) local_root 인자 제거
def create_visual_prompts_denormalized(json_path, json_prefix, output_dir, output_json_path):
    """
    JSON 파일을 읽어 원본 이미지에 바운딩 박스를 그린 후,
    "새 경로"에 새 이미지 파일과 "새 경로"가 적용된 JSON 파일을 저장합니다.
    """
    
    # 1. 입력 JSON 파일 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"성공: '{json_path}' 파일을 읽었습니다.")
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{json_path}'를 찾을 수 없습니다. 경로를 확인하세요.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{json_path}' 파일의 JSON 형식이 올바르지 않습니다.")
        return

    # 2. 출력 폴더 생성 (없으면)
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 폴더: '{output_dir}'")

    # 새 JSON 데이터를 담을 딕셔너리
    new_json_data = {}

    processed_count = 0
    skipped_count = 0

    # 3. JSON 데이터 순회 (key: image_path, value: annotations_list)
    for image_path_in_json, annotations_list in input_data.items():

        if not annotations_list:
            skipped_count += 1
            continue

        # 4. (수정됨) 실제 로컬 이미지 경로는 JSON에 있는 절대 경로를 그대로 사용
        local_image_path = image_path_in_json
        
        # 5. (수정됨) 출력용 상대 경로 계산
        # 예: image_path_in_json = "/DATA/toptoon/Target_paired/강철의 사장님들/3/image.png"
        #     json_prefix = "/DATA/toptoon/Target_paired/"
        #     relative_path = "강철의 사장님들/3/image.png"
        relative_path = image_path_in_json.replace(json_prefix, "", 1)
        
        # 6. (수정됨) 최종 출력 이미지 경로 계산
        # 예: output_dir = "/DATA/toptoon/Target_paired_visual_prompting"
        #     output_image_path = "/DATA/toptoon/Target_paired_visual_prompting/강철의 사장님들/3/image.png"
        output_image_path = os.path.join(output_dir, relative_path)
        output_image_dir = os.path.dirname(output_image_path)
        os.makedirs(output_image_dir, exist_ok=True)

        try:
            # 7. 원본 이미지 열기 (JSON의 절대 경로 사용)
            with Image.open(local_image_path) as img:
                draw = ImageDraw.Draw(img)
                
                # *** (중요) 이미지의 실제 너비와 높이 가져오기 ***
                img_width, img_height = img.size

                # 8. 모든 바운딩 박스 그리기
                for annotation in annotations_list:
                    bbox = annotation.get("bbox_pixel")
                    if bbox:
                        # 9. 정규화된 좌표(0-100)를 픽셀 좌표로 변환
                        norm_x1 = bbox['x1']
                        norm_y1 = bbox['y1']
                        norm_x2 = bbox['x2']
                        norm_y2 = bbox['y2']

                        pixel_x1 = (norm_x1 / 100.0) * img_width
                        pixel_y1 = (norm_y1 / 100.0) * img_height
                        pixel_x2 = (norm_x2 / 100.0) * img_width
                        pixel_y2 = (norm_y2 / 100.0) * img_height
                        
                        # 10. 픽셀 좌표로 사각형 그리기
                        draw.rectangle(
                            [pixel_x1, pixel_y1, pixel_x2, pixel_y2], 
                            outline=BOX_COLOR, 
                            width=BOX_WIDTH
                        )
                
                # 11. 바운딩 박스가 그려진 새 이미지 저장 (새 경로)
                img.save(output_image_path)
                processed_count += 1

                # 12. 새 JSON 데이터에 (새 경로, 주석) 쌍 저장
                new_json_data[output_image_path] = annotations_list

        except FileNotFoundError:
            print(f"경고: 원본 파일을 찾을 수 없습니다. 건너뜁니다: '{local_image_path}'")
            skipped_count += 1
        except Exception as e:
            print(f"오류: '{local_image_path}' 처리 중 문제 발생: {e}")
            skipped_count += 1

    # 13. 루프 종료 후, 수집된 새 JSON 데이터를 파일로 저장
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_json_data, f, ensure_ascii=False, indent=2)
        print(f"\n성공: 새 주석 파일이 '{output_json_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"\n오류: 새 주석 파일 '{output_json_path}' 저장 중 문제 발생: {e}")

    print("\n--- 작업 완료 ---")
    print(f"성공 (이미지 생성): {processed_count} 개")
    print(f"건너뜀 (주석 없음/파일 오류): {skipped_count} 개")
    print(f"새 JSON에 저장된 항목 수: {len(new_json_data)} 개")


# --- (수정됨) 메인 함수 실행 ---
# (테스트용 더미 코드 부분은 원본에서 주석 처리되어 있었으므로, 여기서는 호출만 남깁니다.)

print("Visual Prompt 이미지 및 새 JSON 파일 생성을 시작합니다...")
create_visual_prompts_denormalized(
    json_path=INPUT_JSON_PATH,
    json_prefix=JSON_PATH_PREFIX,  # (수정됨)
    output_dir=OUTPUT_DIR,
    output_json_path=OUTPUT_JSON_PATH
)