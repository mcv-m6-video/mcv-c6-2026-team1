SAVE_DIR=./../SoccerNet/SN-BAS-2025-SAVE_DIR

# Create the folder if it doesn't exist
mkdir -p ${SAVE_DIR}/best

# Download directly to the destination
gdown 1PKIUbJq1g7qF2XThT6AzHcjCrEtijbka -O ${SAVE_DIR}/best/checkpoint_best.pt