#!/bin/bash
#SBATCH --job-name=DSMIL_SimCLR
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s327313@studenti.polito.it
#SBATCH --output=DSMIL_SimCLR.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a40_ext
#SBATCH --mem=32G

# DSMIL + SimCLR Setup and Training Pipeline (Server Edition)

# set -e  # Exit on any error

# cd /home/liannello/MLIAProject/MLIAProject/DSMIL/dsmil-wsi

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate dsmil


# echo "📥 Downloading dataset..."
# wget -O datasetWSI.zip "https://zenodo.org/records/15700269/files/datasetWSI.zip?download=1"
# mkdir -p datasetWSI
# unzip datasetWSI.zip -d datasetWSI

# echo "📁 Organizing dataset directory..."
# mv datasetWSI WSI

# echo "🧩 Running patch extraction..."
# # MPLBACKEND=Agg conda run -n dsmil python deepzoom_tiler.py --magnifications 0 1 -b 10 -d ndpi_files --workers 2 --slide_format ndpi --tile_size 256

# echo "🧹 Cleaning up GPU memory..."
# python -c "
# import torch, gc
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()
# gc.collect()
# "

# echo "⚙️ Running SimCLR training..."
# cd simclr
# MPLBACKEND=Agg conda run -n dsmil python run.py --dataset=ndpi_files --multiscale=1 --level=low
# MPLBACKEND=Agg conda run -n dsmil python run.py --dataset=ndpi_files --multiscale=1 --level=high

# cd ..

# echo "🔍 Computing WSI features..."
# MPLBACKEND=Agg conda run -n dsmil python compute_feats.py --dataset=ndpi_files --magnification=tree \
#     --weights_low=low \
#     --weights_high=high

# echo "📈 Inspecting CSV file shapes:"
# for f in /home/liannello/MLIAProject/MLIAProject/DSMIL/dsmil-wsi/datasets/ndpi_files/*/*.csv; do
#     echo -n "$f: "
#     python -c "import pandas as pd; df = pd.read_csv('$f'); print(df.shape)"
# done

# echo "🎯 Training DSMIL model on extracted features..."
# MPLBACKEND=Agg conda run -n dsmil python train_tcga.py \
#     --dataset=ndpi_files \
#     --num_classes=3 \
#     --feats_size=4096 \
#     --num_epochs=100\
#     --dropout_patch=0.8 \
#     --lr=1e-4 \
#     --weight_decay=1e-5 \
#     --stop_epochs=100 \
#     --eval_scheme=single-split \
#     --split=0.3
# echo "✅ All steps completed successfully."


set -e

cd /home/liannello/MLIAProject/MLIAProject/DSMIL/dsmil-wsi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsmil

# Hyperparameter grids
num_epochs_list=(100 200 250 300 350 400 500 600 )
dropout_patch_list=(0.3 0.5 0.6 0.8 1.0)
lr_list=(1e-4 5e-5 1e-5 )
weight_decay_list=(1e-5 1e-4)

for num_epochs in "${num_epochs_list[@]}"; do
  for dropout_patch in "${dropout_patch_list[@]}"; do
    for lr in "${lr_list[@]}"; do
      for weight_decay in "${weight_decay_list[@]}"; do

        out_log="DSMIL_np${num_epochs}_dp${dropout_patch}_lr${lr}_wd${weight_decay}.log"
        results_csv="results_np${num_epochs}_dp${dropout_patch}_lr${lr}_wd${weight_decay}.csv"

        echo "=== Running: epochs=$num_epochs, dropout_patch=$dropout_patch, lr=$lr, wd=$weight_decay ===" | tee "$out_log"

        MPLBACKEND=Agg conda run -n dsmil python train_tcga.py \
          --dataset=ndpi_files \
          --num_classes=3 \
          --feats_size=4096 \
          --num_epochs=$num_epochs \
          --dropout_patch=$dropout_patch \
          --lr=$lr \
          --weight_decay=$weight_decay \
          --stop_epochs=$num_epochs \
          --eval_scheme=single-split \
          --split=0.3 \
          --results_csv="$results_csv" | tee -a "$out_log"

        echo "✅ Finished: epochs=$num_epochs, dropout_patch=$dropout_patch, lr=$lr, wd=$weight_decay" | tee -a "$out_log"
        echo "-------------------------------------------------------------" | tee -a "$out_log"

      done
    done
  done
done

echo "🏁 All parameter sweeps completed!"