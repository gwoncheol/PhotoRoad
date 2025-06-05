# 파일 이름: convert_image_descriptions.py

def convert_format(input_file, output_file):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with open(output_file, "w", encoding="utf-8") as f:
            for line in lines:
                if ':' in line:
                    filename, description = line.strip().split(':', 1)
                    filename = filename.strip()
                    description = description.strip()
                    f.write(f'{filename}, "{description}"\n')
        print(f"변환이 완료되었습니다. 결과는 '{output_file}'에 저장되었습니다.")
    except FileNotFoundError:
        print(f"입력 파일 '{input_file}'을(를) 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 사용 예시
if __name__ == "__main__":
    input_path = "blip2_captions2.txt"      # 원본 파일 경로 (예: 이미지이름:설명)
    output_path = "output.txt"    # 변환된 결과 저장 파일
    convert_format(input_path, output_path)
