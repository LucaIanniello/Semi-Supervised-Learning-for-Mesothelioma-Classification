#!/bin/bash
#SBATCH --job-name=TrainingCLAM
#SBATCH --mem=20000M
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s327313@studenti.polito.it
#SBATCH --output=execution.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a40_ext

set -e
echo "🚀 Starting CLAM training..."

conda init
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clam_latest



# Download dataset
cd /home/liannello/MLIAProject/MLIAProject/CLAM


# Create data splits
export MPLBACKEND=Agg
conda run -n clam_latest python /home/liannello/MLIAProject/MLIAProject/CLAM/create_splits_seq.py \
  --task MLIA_Project --seed 1 --k 1 --val_frac 0.2 --test_frac 0.2

# Train model
CUDA_VISIBLE_DEVICES=0 conda run -n clam_latest python /home/liannello/MLIAProject/MLIAProject/CLAM/main.py \
  --max_epoch 300 \
  --drop_out 0.25 \
  --lr 1e-4 \
  --k 1 \
  --exp_code MLIA_CLAM_50 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task MLIA_Project \
  --model_type clam_sb \
  --log_data \
  --subtyping \
  --data_root_dir ./results_features \
  --embed_dim 1024

# Zip training results
zip -r result_classifier.zip results/

# Evaluate model
CUDA_VISIBLE_DEVICES=0 conda run -n clam_latest python /home/liannello/MLIAProject/MLIAProject/CLAM/eval.py \
  --k 10 \
  --models_exp_code MLIA_CLAM_50_s1 \
  --save_exp_code MLIA_CLAM_eval \
  --task MLIA_Project \
  --model_type clam_sb \
  --results_dir results \
  --data_root_dir ./results_features \
  --embed_dim 1024

# Zip evaluation results
zip -r result_evaluation.zip eval_results/

echo "✅ CLAM Training + Evaluation completed!"
