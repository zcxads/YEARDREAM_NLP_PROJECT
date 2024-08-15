# 한국어 일반 상식 객관식 문제 풀이

## 문제 정의
- 다양한 도메인의 상식에 대한 객관식 문제의 정답을 맞추는 LLM 모델 개발
- LLM 학습, 튜닝, 프롬프트 엔지니어링 등과 LLM 활용 데이터 증강 등의 다양한 경험

## 데이터 설명
- 컬럼 구조는 아래와 같음
- test 데이터에는 'id'컬럼이 추가적으로 존재하고, '답안' 컬럼은 존재하지 않음
- 답안은 선택지의 인덱스로, 0부터 3사이의 값을 가짐

![image](https://github.com/user-attachments/assets/7c8499fe-3c92-49e4-930b-6d39cdec1206)

## 평가지표
- Accuracy score : 전체 질문 중 정답을 맞힌 질문의 비율

## Directory Structure
```
├── data/
│   ├── train_data.csv
│   └── test_data.csv
│
├── configs/
│   └── config.yaml
│
├── utils.py
├── train.py
├── inference.py
├── run.sh
└── sample_submission.csv
```

## Description
- `data/`: 학습데이터/평가데이터 csv 파일 
- `configs/`: accelerate 및 fsdp config 파일
- `utils.py`: 학습 및 추론하는데 필요한 함수들 정리한 python script
- `train.py`: LLM 모델 학습하는 python script
- `inference.py`: LLM 모델 추론하는 python script
- `run.sh`: LLM 모델 config 값 설정하여 train.py 실행하는 shell script
- `sample_submission.csv`: 제출 csv 파일 예시

## Getting Started
- 도커 컨테이너 접속 : 
    - 로컬 호스트에서 로컬 포트를 원격 호스트로 포워딩 
        `ssh -i ~/.ssh/student_keypair.pem -L 2222:localhost:2222 ubuntu@원격_호스트_IP`
    - 새로운 셸을 열어 포트를 지정해주는 방식으로 ssh 연결
        `ssh -p 2222 root:localhost`
- 환경 설정 : `cat requirements.txt | xargs -n 1 pip install`
- 학습 : config 수정 후, `sh run.sh` 명령어 실행 
- 추론 : `CUDA_VISIBLE_DEVICES=0 python3 inference.py --model_path MODEL_PATH`
  (MODEL_PATH = results/...)
