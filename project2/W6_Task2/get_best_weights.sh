SAVE_DIR=./../SoccerNet/SN-BAS-2025-SAVE_DIR

# Create the folders if they do not exist
mkdir -p ${SAVE_DIR}/w6_best
mkdir -p ${SAVE_DIR}/best

# Download directly to the destination
gdown 1TSnRadcNDnruvL7XQTCjRgWlxKfLnfWa -O ${SAVE_DIR}/w6_best/checkpoint_best.pt
gdown 1JU9XTJZElpXA0MqmQNk1_OCBjukktDOW -O ${SAVE_DIR}/best/checkpoint_best.pt
