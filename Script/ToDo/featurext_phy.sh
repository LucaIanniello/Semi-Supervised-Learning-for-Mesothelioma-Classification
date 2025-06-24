#!/bin/bash
#SBATCH --job-name=TridentFeatureExtraction
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s327313@studenti.polito.it
#SBATCH --output=trident.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a40_ext
#SBATCH --mem=32G
export OMP_NUM_THREADS=2

set -e

echo "Starting Trident feature extraction job..."

# Variables
ACCESS_TOKEN='uVSb7icJqT9efPM71KYgviJ50r7eML9ynei2q7hDkedVlFrf8fBsr9lFaJ3O'
DATASET_URL="https://zenodo.org/records/15700269/files/datasetWSI.zip?download=1"
CONDA_ENV="trident"

# Setup
cd /home/liannello/MLIAProject/MLIAProject/Legion

# # Download and unzip dataset
# wget -O Train.zip "$DATASET_URL"
# unzip -o Train.zip -d .

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create and activate env if not existing
if ! conda info --envs | grep -q "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" python=3.10
fi
conda activate "$CONDA_ENV"

# Clone Trident
if [ ! -d trident ]; then
  git clone https://github.com/mahmoodlab/trident.git
  conda run -n trident pip install -e trident/.
fi

# Run Trident on all folders
export MPLBACKEND=Agg

for folder in B E S; do
  conda run -n trident python trident/run_batch_of_slides.py \
    --task all \
    --wsi_dir "./ndpi_files/${folder}" \
    --job_dir "./trident_processed_phikon/${folder}" \
    --patch_encoder phikon_v2 \
    --patch_size 224 \
    --mag 20 \
    --num_workers 2
done

# Create zip archive
ZIP_NAME="datasetTrident_phikon.zip"
echo "Creating zip archive: $ZIP_NAME"
zip -r "$ZIP_NAME" trident_processed_phikon

# Upload to Zenodo
echo "Uploading to Zenodo..."
python3 <<EOF
import requests
import json
import os

ACCESS_TOKEN = "${ACCESS_TOKEN}"
def create_deposition(title):
    url = 'https://zenodo.org/api/deposit/depositions'
    headers = {"Content-Type": "application/json"}
    params = {'access_token': ACCESS_TOKEN}
    data = {
        'metadata': {
            'title': title,
            'upload_type': 'dataset',
            'description': 'Dataset WSI project MLiA',
            'creators': [{'name': 'Raf-Tony-Luca'}]
        }
    }
    r = requests.post(url, params=params, data=json.dumps(data), headers=headers)
    return r.json()

def upload_file(deposition_id, file_path):
    url = f'https://zenodo.org/api/deposit/depositions/{deposition_id}'
    params = {'access_token': ACCESS_TOKEN}
    r = requests.get(url, params=params)
    bucket_url = r.json()["links"]["bucket"]
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as fp:
        r = requests.put(f"{bucket_url}/{filename}", data=fp, params=params)
    return r.json()

def publish_deposition(deposition_id):
    url = f'https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish'
    params = {'access_token': ACCESS_TOKEN}
    r = requests.post(url, params=params)
    return r.json()

deposition = create_deposition("dataset_trident_phikon")
deposition_id = deposition['id']
print(f"Uploading file... (ID: {deposition_id})")
upload_result = upload_file(deposition_id, "$ZIP_NAME")
print("Publishing dataset...")
publication = publish_deposition(deposition_id)
print(f"Dataset published! DOI: {publication['doi']}")
print(f"URL: {publication['links']['record_html']}")
EOF

echo "Job completed!"
