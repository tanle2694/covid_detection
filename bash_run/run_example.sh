set -e

ROOT_FOLDER=""
EXP_NAME=""
TRAIN_DATA=""
VALIDATION_DATA=""
TRAIN_SAVE_DIR=""

python train_v2.py --train_data $TRAIN_DATA \
                   --validation_data $VALIDATION_DATA \
                   --root_folder $ROOT_FOLDER \
                   --exper_name $EXP_NAME \
                   --trainer_save_dir $TRAIN_SAVE_DIR
