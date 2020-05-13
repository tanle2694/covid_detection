set -e

VALIDATION_DATA=""
ROOT_FOLDER=""
DEVICE=""
PREDICT_RESULT=""
RESUME=""


python inference_images.py --validation_data $VALIDATION_DATA \
                           --root_folder $ROOT_FOLDER \
                           --device $DEVICE \
                           --predict_result $PREDICT_RESULT \
                           --resume $RESUME
