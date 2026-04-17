SAVE_DIR=./../SoccerNet/SN-BAS-2025-SAVE_DIR

# Create the folder if it doesn't exist
mkdir -p ${SAVE_DIR}/best

# Download directly to the destination
gdown 1-I4XEijAlISe8Tz_fqM0UK9LVDskcMG3 -O ${SAVE_DIR}/best/checkpoint_best.pt