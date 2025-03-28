import cv2
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from rest_framework.decorators import api_view
from rest_framework.response import Response

# 텐서플로 라이트 모델 로딩
interpreter = tflite.Interpreter(model_path="./1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 텍스트 인식 함수
def recognize_text_with_tflite(image_np):
    # 모델 입력 크기 가져오기
    input_shape = input_details[0]['shape']  # e.g. (1, 320, 320, 3)
    height, width = input_shape[1], input_shape[2]

    # 이미지 리사이즈 및 전처리
    resized = cv2.resize(image_np, (width, height))
    input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    # 추론 실행
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 출력 텐서에서 결과 추출
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data  # 디코딩 필요

# API 뷰
@api_view(['POST'])
def detect_text(request):
    image_file = request.FILES.get('image')
    query = request.data.get('query', '').strip().lower()

    if not image_file or not query:
        return Response({'error': '이미지와 검색어를 보내주세요.'}, status=400)

    try:
        # PIL → NumPy
        image = Image.open(image_file).convert('RGB')
        image_np = np.array(image)

        # 텍스트 인식 수행
        predictions = recognize_text_with_tflite(image_np)

        # 디코딩 및 필터링 (예: 문자열, 위치 정보는 모델에 따라 다름)
        # 아래는 예시 구조이며, 사용하는 모델 구조에 맞게 수정 필요
        matches = []
        for item in predictions:  # 예: 각 item = {"text": "단어", "bbox": [...], "score": ...}
            text = item['text'].strip()
            if query in text.lower():
                matches.append(item)

        return Response({'matches': matches})

    except Exception as e:
        return Response({'error': str(e)}, status=500)
