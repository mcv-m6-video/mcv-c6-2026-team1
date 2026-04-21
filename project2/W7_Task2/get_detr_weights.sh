SAVE_DIR=./../SoccerNet/SN-BAS-2025-SAVE_DIR

# Create the folder if it doesn't exist
mkdir -p ${SAVE_DIR}/detr

# Download directly to the destination
gdown 1XVGfOR4KZ3Bgjyb4pv1KexJZ8qUELact -O ${SAVE_DIR}/detr/checkpoint_best.pt