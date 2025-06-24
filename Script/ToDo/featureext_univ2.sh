#!/bin/bash
#SBATCH --job-name=TridentFeatureExtraction
#SBATCH --mem=20000M
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s327313@studenti.polito.it
#SBATCH --output=trident_2.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a40_ext
export OMP_NUM_THREADS=2

set -e

echo "Starting Trident Univ2 feature extraction..."

# Set env var for PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg

# Vars
ACCESS_TOKEN='uVSb7icJqT9efPM71KYgviJ50r7eML9ynei2q7hDkedVlFrf8fBsr9lFaJ3O'
HF_TOKEN=''
DATASET_URL="https://zenodo.org/records/15700269/files/datasetWSI.zip?download=1"
CONDA_ENV="trident"

# Setup directory
cd /home/liannello/MLIAProject/MLIAProject/Legion

# Download dataset
# wget -O Train.zip "$DATASET_URL"
# unzip -o Train.zip -d .

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create and activate env
# if ! conda info --envs | grep -q "$CONDA_ENV"; then
#   conda create -y -n "$CONDA_ENV" python=3.10
# fi
conda activate "$CONDA_ENV"

# Login to Hugging Face
python3 -c "from huggingface_hub import login; login('${HF_TOKEN}')"

# # Install Trident
# if [ ! -d trident ]; then
#   git clone https://github.com/mahmoodlab/trident.git
#   conda run -n trident pip install -e trident/.
# fi

# Free memory function
function free_memory() {
  python3 -c "import torch, gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()"
}

# Run Trident
for folder in B E S; do
  conda run -n trident python trident/run_batch_of_slides.py \
    --task all \
    --wsi_dir "./ndpi_files/${folder}" \
    --job_dir "./trident_processed_univ2/${folder}" \
    --patch_encoder uni_v2 \
    --patch_size 256 \
    --mag 20 \
    --max_workers 2
done

# Create ZIP
ZIP_NAME="datasetTrident_univ2.zip"
zip -r "$ZIP_NAME" trident_processed_univ2

# Upload script
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

deposition = create_deposition("dataset_trident_univ2")
deposition_id = deposition['id']
print(f"Uploading file... (ID: {deposition_id})")
upload_result = upload_file(deposition_id, "${ZIP_NAME}")
publication = publish_deposition(deposition_id)

print(f"Dataset published! DOI: {publication['doi']}")
print(f"URL: {publication['links']['record_html']}")
EOF

echo "✅ Done!"
