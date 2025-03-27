from rest_framework.decorators import api_view
from rest_framework.response import Response
import easyocr
import numpy as np
from PIL import Image

# EasyOCR 리더 객체 생성 ('ko'는 한글, 'en'은 영어 인식 설정, GPU 사용 안 함)
reader = easyocr.Reader(['ko', 'en'], gpu=False)  # GPU 없으면 False

# REST API에서 POST 요청만 허용하는 뷰 함수 정의
@api_view(['POST'])
def detect_text(request):
    # 클라이언트가 보낸 파일 중에서 'image'라는 키로 업로드된 이미지 파일을 가져옴
    image_file = request.FILES.get('image')

    # 클라이언트가 보낸 문자열 'query' (찾고자 하는 단어)를 소문자로 변환하고 앞뒤 공백 제거
    query = request.data.get('query', '').strip().lower()

    # 이미지나 검색어가 누락된 경우, 에러 응답 반환 (HTTP 상태 코드 400: Bad Request)
    if not image_file or not query:
        return Response({'error': '이미지와 검색어를 보내주세요.'}, status=400)

    # 업로드된 이미지를 열고 RGB 포맷으로 변환 (PIL 객체)
    image = Image.open(image_file).convert('RGB')

    # PIL 이미지를 NumPy 배열로 변환 (OpenCV나 EasyOCR에서 사용하기 위함)
    image_np = np.array(image)

    # EasyOCR을 사용하여 이미지에서 텍스트 인식 실행
    # 결과는 [(bbox, text, prob), ...] 형태의 리스트로 반환됨
    results = reader.readtext(image_np)

    # 검색어와 일치하는 텍스트 결과를 담을 리스트 초기화
    matches = []

    # OCR 결과 중 하나씩 반복
    for (bbox, text, prob) in results:
        # 인식된 텍스트를 소문자로 바꾼 후, 검색어가 포함되어 있는지 확인
        if query in text.lower():
            # 검색어가 포함된 텍스트의 정보 (텍스트 내용, 경계 상자 좌표, 신뢰도)를 matches에 추가
            matches.append({
                'text': text,        # 인식된 텍스트
                'bbox': bbox,        # 텍스트가 위치한 박스 좌표 (4개의 점 좌표)
                'prob': round(prob, 3)  # 인식 확률 (소수점 셋째 자리까지 반올림)
            })

    # 최종 결과를 JSON 형식으로 응답 (검색어가 포함된 텍스트만)
    return Response({'matches': matches})
