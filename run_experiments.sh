#!/bin/bash


#실행방법
#chmod +x run_experiments.sh  # 실행 권한 부여
#./run_experiments.sh         # 실행

# CSV 파일 리스트
CSV_FILES=($(ls ./dataset/label/train/*.csv))
VAL_FILES=($(ls ./dataset/label//val/*.csv))
TEST_FILES=($(ls ./dataset/label/test/*.csv))

# 공통 설정
MODEL_NAME="vit_small_patch16_224"
BATCH_SIZE=32
LR=0.001
EPOCHS=10
NUM_CLASSES=6

# 반복적으로 Python 스크립트를 실행
for i in "${!CSV_FILES[@]}"; do
    TRAIN_CSV=${CSV_FILES[$i]}
    VAL_CSV=${VAL_FILES[$i]}
    TEST_CSV=${TEST_FILES[$i]}

    # 실험 이름 설정
    EXPERIMENT_NAME="experiment_$((i+1))"

    # Python 스크립트 실행
    python script.py --name $EXPERIMENT_NAME \
                     --model_name $MODEL_NAME \
                     --train_data $TRAIN_CSV \
                     --val_data $VAL_CSV \
                     --test_data $TEST_CSV \
                     --batch_size $BATCH_SIZE \
                     --lr $LR \
                     --epochs $EPOCHS \
                     --num_classes $NUM_CLASSES
done
