#폴더 안에 있는 이미지 이름으로 csv 파일 생성하는 코드
import os
import csv

# 이미지가 있는 폴더 경로 지정
folder_path = "C:/Users/dlxlr/Desktop/JejuImage"

# 이미지 파일 확장자 목록 (필요시 수정 가능)
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# CSV 파일을 작성할 경로 지정
csv_file_path = 'output.csv'

# 이미지 파일 목록을 CSV로 작성하는 함수
def write_image_names_to_csv(folder_path, csv_file_path):
    # 폴더 내 모든 파일 탐색
    image_files = [f for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # CSV 파일 작성
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name'])  # CSV 첫 번째 행에 헤더 작성
        
        # 이미지 파일 이름을 한 줄씩 작성
        for image in image_files:
            writer.writerow([image])
    
    print(f'{len(image_files)}개의 이미지 파일을 {csv_file_path}에 기록했습니다.')

# 함수 실행
write_image_names_to_csv(folder_path, csv_file_path)
