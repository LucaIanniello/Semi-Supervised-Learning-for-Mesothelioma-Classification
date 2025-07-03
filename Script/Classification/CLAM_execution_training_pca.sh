#!/bin/bash
#SBATCH --job-name=CLAMCLassificationPCA
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s327313@studenti.polito.it
#SBATCH --output=clam_classification_pca_focal.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a40_ext
#SBATCH --mem=32G

set -e
echo "=== Starting CLAM multi-extractor training ==="

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clam_latest

# Root paths
BASE_DIR=/home/liannello/MLIAProject/MLIAProject
CLAM_DIR=$BASE_DIR/CLAM
FEATURE_DIR=$CLAM_DIR/results_features/pt_files
mkdir -p "$FEATURE_DIR"

# List of feature extractors
extractors=(
    "augdiff-resnet50-trident"
)

for feature_extractor in "${extractors[@]}"; do
    echo ">>> Processing $feature_extractor"

    # Create working dir
    WORK_DIR=$BASE_DIR/CLAM_RUNS_PCA_WCE/$feature_extractor
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"

    # Select Zenodo URL
    case $feature_extractor in
        ("augdiff-resnet50-trident")
            url="https://zenodo.org/records/15786844/files/datasetCompleted.zip?download=1"
            extract_path="./results_features/pt_files"
            ;;
        (*)
            echo "Unknown feature extractor: $feature_extractor"
            exit 1
            ;;
    esac

    # Download and unzip
    echo ">> Downloading dataset for $feature_extractor"
    wget -O Train.zip "$url"
    mkdir -p "$extract_path"
    unzip -o Train.zip -d "$extract_path"

    echo ">> Converting .h5 to .pt and creating CSV for trident extractors..."
    pca_dim=$(python3 /home/liannello/MLIAProject/convert_h5_to_pt_and_csv.py --feature_extractor "$feature_extractor" --output_dir "$WORK_DIR/results_features/pt_files/" --pca | tail -1)
    echo "PCA output dimension: $pca_dim"

    echo ">> Creating splits"
    MPLBACKEND=Agg conda run -n clam_latest python $CLAM_DIR/create_splits_seq.py \
        --task MLIA_Project --seed 1 --k 1 --val_frac 0.2 --test_frac 0.2 \
        --feature_extractor "$feature_extractor"

    echo ">> Training CLAM"
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 conda run -n clam_latest python $CLAM_DIR/main.py \
        --max_epoch 300 --drop_out 0.25 --lr 1e-4 --k 1 --exp_code "CLAM_${feature_extractor}_50" \
        --weighted_sample --bag_loss w_ce --inst_loss svm --task MLIA_Project \
        --model_type clam_sb --log_data --subtyping \
        --data_root_dir "$WORK_DIR/results_features" --embed_dim $pca_dim \
        --feature_extractor "$feature_extractor"

    echo ">> Zipping training results"
    zip -r "results_${feature_extractor}.zip" "$WORK_DIR/results"

    echo ">> Evaluating CLAM"
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 conda run -n clam_latest python $CLAM_DIR/eval.py \
        --k 1 --models_exp_code "CLAM_${feature_extractor}_50_s1" \
        --save_exp_code "CLAM_eval_${feature_extractor}" \
        --task MLIA_Project --model_type clam_sb --results_dir "$WORK_DIR/results" \
        --data_root_dir "$WORK_DIR/results_features" --embed_dim $pca_dim\
        --feature_extractor "$feature_extractor"

    echo ">> Zipping evaluation results"
    zip -r "eval_${feature_extractor}.zip" "$WORK_DIR/eval_results"

    echo "✅ Done with $feature_extractor"
done

echo "🏁 All feature extractors completed!"
