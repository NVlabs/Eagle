set -a
source .env
set +a

RECIPE_PATH=${1:-"local_playground/recipe/stage1.json"}
NODES=${2:-1}
LOG_DIR=${3:-"work_dirs/data_prepare"}
TOKENIZER=${4:-"Qwen/Qwen3-1.7B"}
LAUNCHER=${5:-"pytorch"}


bash shell/prepare.sh ${RECIPE_PATH} ${NODES} ${LOG_DIR} ${TOKENIZER} ${LAUNCHER}

# submit_job \
#     --image=${TRAINING_IMAGE_PATH} \
#     --gpu 8 \
#     --tasks_per_node 8 \
#     --nodes ${NODES} \
#     -n prepare_data \
#     --logroot ${LOG_DIR} \
#     --email_mode never \
#     --duration 0 \
#     --cpu 128 \
#     --dependent_clones 3 \
#     --partition adlr_services \
#     -c "bash shell/eagle_abc/prepare.sh  ${RECIPE_PATH} ${NODES} ${LOG_DIR} ${TOKENIZER} ${LAUNCHER}"
