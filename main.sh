
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

DATASET=profession-cat
LABEL_NAME_FILE=label_names.txt
TRAIN_CORPUS=train.txt
TEST_CORPUS=test.txt
TEST_LABEL=test_labels.txt
#MAP_CORPUS=map.txt
GOLD_CORPUS=gold_labels.txt
WEB_CORPUS=web.txt
WEB_LABELS=web_labels.txt
MAX_LEN=512
TRAIN_BATCH=4
ACCUM_STEP=16
EVAL_BATCH=8
GPUS=2
MCP_EPOCH=100
SELF_TRAIN_EPOCH=3

python src/train.py --dataset_dir datasets/${DATASET}/ --label_names_file ${LABEL_NAME_FILE} \
                    --train_file ${TRAIN_CORPUS} \
                    --web_file ${WEB_CORPUS} --web_label_file ${WEB_LABELS} \
                    --gold_label_file ${GOLD_CORPUS} \
                    --test_file ${TEST_CORPUS} --test_label_file ${TEST_LABEL} \
                    --max_len ${MAX_LEN} \
                    --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                    --gpus ${GPUS} \
                    --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} \
