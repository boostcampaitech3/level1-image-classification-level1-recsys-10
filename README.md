# pstage_01_image_classification

## Getting Started

### Dependencies

- torch==1.6.0
- torchvision==0.7.0

### Install Requirements

- `pip install -r requirements.txt`
    - library 추가 : 개인이 추가할 것

---

### Data pre-processing

1. correct_mislabeled_data.py
    - 주어진 데이터에서 mislabeled된 데이터 수정
    `python correct_mislabeled_data.py`
2. face_detection.py
    - train data에서, face detection을 통해 image cropping
    `python face_detection.py —data_type train`
    - eval data에서, face detection을 통해 image cropping
    `python face_detection.py —data_type eval`
3. cycleGAN으로 fake image 생성
    1. face_detection을 통해 만들어진 image를 이용해 fake image 생성한 데이터
    2. 원본 데이터를 모두 cycleGAN에 넣어 fake image 생성

---

### Training

- fake image directory를 training data로 설정
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference

- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation

- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`