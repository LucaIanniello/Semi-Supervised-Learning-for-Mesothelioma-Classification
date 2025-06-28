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

set -e  # Exit on any error

echo "🧬 Cloning DSMIL repository..."
git clone https://github.com/binli123/dsmil-wsi.git
cd dsmil-wsi

echo "📦 Creating Conda environment..."
conda env create -f env.yml
conda activate dsmil

echo "🔧 Installing required Python packages..."
pip install torch torchvision openslide-python openslide-bin tensorboard pandas

echo "📥 Downloading dataset..."
wget -O datasetWSI.zip "https://zenodo.org/records/15700269/files/datasetWSI.zip?download=1"
mkdir -p datasetWSI
unzip datasetWSI.zip -d datasetWSI

echo "📁 Organizing dataset directory..."
mv datasetWSI WSI

echo "🧩 Running patch extraction..."
MPLBACKEND=Agg conda run -n dsmil python deepzoom_tiler.py --magnifications 0 1 -b 10 -d ndpi_files --workers 4 --slide_format ndpi --tile_size 256

echo "🧹 Cleaning up GPU memory..."
python -c "
import torch, gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()
"

echo "✏️ Overwriting simclr/config.yaml with the content of SIMCLR_feature_extraction_utils/config.yaml..."
cat > simclr/config.yaml << EOF
batch_size: 64
epochs: 20
eval_every_n_epochs: 5
fine_tune_from: ''
log_every_n_steps: 20
weight_decay: 10e-6
fp16_precision: False
n_gpu: 1
gpu_ids: [0]

model:
  out_dim: 256
  base_model: "resnet18"

dataset:
  s: 1
  input_shape: (256,256,3)
  num_workers: 2
  valid_size: 0.1

loss:
  temperature: 0.5
  use_cosine_similarity: True
EOF

echo "✏️ Overwriting simclr/run.py with the content of SIMCLR_feature_extraction_utils/run.py..."
cat > simclr/run.py << EOF
from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse

def generate_csv(args):
    if args.level=='high' and args.multiscale==1:
        path_temp = os.path.join('..', 'WSI', args.dataset, 'pyramid', '*', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/5x_name/*.jpeg
    if args.level=='low' and args.multiscale==1:
        path_temp = os.path.join('..', 'WSI', args.dataset, 'pyramid', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    if args.multiscale==0:
        path_temp = os.path.join('..', 'WSI', args.dataset, 'single', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    df = pd.DataFrame(patch_path)
    df.to_csv('all_patches.csv', index=False)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='low', help='Magnification level to compute embedder (low/high)')
    parser.add_argument('--multiscale', type=int, default=0, help='Whether the patches are cropped from multiscale (0/1-no/yes)')
    parser.add_argument('--dataset', type=str, default='TCGA-lung', help='Dataset folder name')
    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    gpu_ids = config['gpu_ids']
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
 
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])   
    generate_csv(args)
    simclr = SimCLR(dataset, config, args)
    simclr.train()


if __name__ == "__main__":
    main()

EOF

echo "✏️ Overwriting simclr/simclr.py with the content of SIMCLR_feature_extraction_utils/simclr.py..."
cat > simclr/simclr.py << EOF
import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config, args):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=f"runs/{args.level}")
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"])# .to(self.device)
        if self.config['n_gpu'] > 1:
            device_n = len(eval(self.config['gpu_ids']))
            model = torch.nn.DataParallel(model, device_ids=range(device_n))
        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)
            

        optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(self.config['weight_decay']))

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
#                                                                last_epoch=-1)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
                                                               last_epoch=-1)
        

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for (xis, xjs) in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print('saved')

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        import torchvision.models as models
        import torch

        print(model)

        try:
            # Load torchvision pretrained resnet18
            pretrained_resnet = models.resnet18(pretrained=True)

            # Assuming your ResNetSimCLR has an attribute 'encoder' or similar where the backbone is:
            # Copy pretrained weights from torchvision model to your model's encoder
            model.features.load_state_dict(pretrained_resnet.state_dict(), strict=False)

            print("Loaded pretrained ResNet-18 weights successfully.")
        except FileNotFoundError:
            print("Pretrained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

EOF




echo "⚙️ Running SimCLR training..."
cd simclr
MPLBACKEND=Agg conda run -n dsmil python run.py --dataset=ndpi_files --multiscale=1 --level=low
MPLBACKEND=Agg conda run -n dsmil python run.py --dataset=ndpi_files --multiscale=1 --level=high

cd ..

echo "🔍 Computing WSI features..."
MPLBACKEND=Agg conda run -n dsmil python compute_feats.py --dataset=ndpi_files --magnification=tree \
    --weights_low=low \
    --weights_high=high

echo "📈 Inspecting CSV file shapes:"
for f in /content/dsmil-wsi/datasets/ndpi_files/*/*.csv; do
    echo -n "$f: "
    python -c "import pandas as pd; df = pd.read_csv('$f'); print(df.shape)"
done






echo "✏️ Overwriting train_tcga.py with content of train_MLiA_SimCLR.py..."
cat > train_tcga.py << EOF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm

def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats, feats_csv_path

def generate_pt_files(args, df):
    temp_train_dir = "temp_train"
    if os.path.exists(temp_train_dir):
        import shutil
        shutil.rmtree(temp_train_dir, ignore_errors=True)
    os.makedirs(temp_train_dir, exist_ok=True)
    print('Creating intermediate training files.')
    for i in tqdm(range(len(df))):
        label, feats, feats_csv_path = get_bag_feats(df.iloc[i], args)
        bag_label = torch.tensor(np.array([label]), dtype=torch.float32)
        bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
        repeated_label = bag_label.repeat(bag_feats.size(0), 1)
        stacked_data = torch.cat((bag_feats, repeated_label), dim=1)
        # Save the stacked data into a .pt file
        pt_file_path = os.path.join(temp_train_dir, os.path.splitext(feats_csv_path)[0].split(os.sep)[-1] + ".pt")
        torch.save(stacked_data, pt_file_path)



def train(args, train_df, milnet, criterion, optimizer):
    milnet.train()
    dirs = shuffle(train_df)
    total_loss = 0
    Tensor = torch.cuda.FloatTensor
    for i, item in enumerate(dirs):
        optimizer.zero_grad()
        stacked_data = torch.load(item, map_location='cuda:0')
        bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
        bag_feats = Tensor(stacked_data[:, :args.feats_size])
        bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows

def test(args, test_df, milnet, criterion, thresholds=None, return_predictions=False):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, item in enumerate(test_df):
            stacked_data = torch.load(item, map_location='cuda:0')
            bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
            bag_feats = Tensor(stacked_data[:, :args.feats_size])
            bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([(torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    if return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            continue
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs):
    if args.dataset.startswith('TCGA-lung'):
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    else:
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

def get_current_score(avg_score, aucs):
    current_score = (sum(aucs) + avg_score)/2
    return current_score

def save_model(args, fold, run, save_path, model, thresholds_optimal):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run+1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)
    file_name = os.path.join(save_path, f'fold_{fold}_{run+1}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)

def print_save_message(args, save_name, thresholds_optimal):
    if args.dataset.startswith('TCGA-lung'):
        print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
    else:
        print('Best model saved at: ' + save_name)
        print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=10, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--eval_scheme', default='5-fold-cv', type=str, help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-time-train+valid+test ]')

    
    args = parser.parse_args()
    print(args.eval_scheme)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    def apply_sparse_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_model(args):
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        milnet.apply(lambda m: apply_sparse_init(m))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        return milnet, criterion, optimizer, scheduler
    
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('/content/dsmil-wsi/datasets/', args.dataset, args.dataset+'.csv')

    generate_pt_files(args, pd.read_csv(bags_csv))

    if args.eval_scheme == '5-fold-cv':

        import csv
        from sklearn.model_selection import StratifiedKFold, train_test_split

        print('5-fold-cv')
        bags_path = glob.glob('temp_train/*.pt')
        kf = StratifiedKFold(n_splits=1, shuffle=True, random_state=42)

        fold_results = []
        summary_rows = []

        def get_class_label(path, feats_size):
            data = torch.load(path)
            label = data[0, feats_size:].cpu().numpy()
            return int(np.argmax(label))  # assumes one-hot encoding

        labels = [get_class_label(p, args.feats_size) for p in bags_path]

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for fold, (train_index, test_index) in enumerate(kf.split(bags_path, labels)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)

            all_train_paths = [bags_path[i] for i in train_index]
            all_train_labels = [labels[i] for i in train_index]

            # Create validation split from training set
            train_path, val_path, train_labels_sub, val_labels_sub = train_test_split(
                all_train_paths,
                all_train_labels,
                test_size=0.3,
                stratify=all_train_labels,
                random_state=fold
            )
            test_path = [bags_path[i] for i in test_index]

            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            best_model = None

            for epoch in range(1, args.num_epochs + 1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer)
                val_loss_bag, val_avg_score, val_aucs, val_thresholds = test(args, val_path, milnet, criterion)

                print_epoch_info(epoch, args, train_loss_bag, val_loss_bag, val_avg_score, val_aucs)
                scheduler.step()

                current_score = get_current_score(val_avg_score, val_aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = val_avg_score
                    best_auc = val_aucs
                    best_model = copy.deepcopy(milnet)
                    save_model(args, fold, run, save_path, milnet, val_thresholds)
                if counter > args.stop_epochs:
                    break

            # Final test on test set using best model
            test_loss_bag, test_avg_score, test_aucs, _ = test(args, test_path, best_model, criterion)

            summary_rows.append({
                "folds": fold,
                "test_auc": float(np.mean(test_aucs)),
                "val_auc": float(np.mean(best_auc)),
                "test_acc": float(test_avg_score),
                "val_acc": float(best_ac)
            })

            fold_results.append((test_avg_score, test_aucs))

        # Save CSV
        csv_path = os.path.join(save_path, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["folds", "test_auc", "val_auc", "test_acc", "val_acc"])
            writer.writeheader()
            writer.writerows(summary_rows)

        # Print final mean results
        mean_ac = np.mean([r["test_acc"] for r in summary_rows])
        mean_auc = np.mean([r["test_auc"] for r in summary_rows])
        print(f"\nFinal Summary:\nMean Test Accuracy: {mean_ac:.4f} | Mean Test AUC: {mean_auc:.4f}")


    elif args.eval_scheme == '5-time-train+valid+test':
        bags_path = glob.glob('temp_train/*.pt')
        # bags_path = bags_path.sample(n=50, random_state=42)
        fold_results = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for iteration in range(5):
            print(f"Starting iteration {iteration + 1}.")
            milnet, criterion, optimizer, scheduler = init_model(args)

            bags_path = shuffle(bags_path)
            total_samples = len(bags_path)
            train_end = int(total_samples * (1-args.split-0.1))
            val_end = train_end + int(total_samples * 0.1)

            train_path = bags_path[:train_end]
            val_path = bags_path[train_end:val_end]
            test_path = bags_path[val_end:]

            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0

            for epoch in range(1, args.num_epochs + 1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer) # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, val_path, milnet, criterion)
                
                print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(args, iteration, run, save_path, milnet, thresholds_optimal)
                    best_model = copy.deepcopy(milnet)
                if counter > args.stop_epochs: break
            test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, best_model, criterion, args)
            fold_results.append((best_ac, best_auc))
        mean_ac = np.mean(np.array([i[0] for i in fold_results]))
        mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
        # Print mean and std deviation for each class
        print(f"Final results: Mean Accuracy: {mean_ac}")
        for i, mean_score in enumerate(mean_auc):
            print(f"Class {i}: Mean AUC = {mean_score:.4f}")


    if args.eval_scheme == 'single-split':
        import csv
        import datetime
        import copy
        from sklearn.model_selection import train_test_split

        print('Single split (train/val/test)')

        bags_path = glob.glob('temp_train/*.pt')

        def get_class_label(path, feats_size):
            data = torch.load(path)
            label = data[0, feats_size:].cpu().numpy()
            return int(np.argmax(label))  # assumes one-hot encoding

        labels = [get_class_label(p, args.feats_size) for p in bags_path]

        # Train/val/test split (60% train, 20% val, 20% test)
        trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
            bags_path, labels, test_size=0.2, stratify=labels, random_state=42
        )
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            trainval_paths, trainval_labels, test_size=0.25, stratify=trainval_labels, random_state=42
        )

        # Save directory
        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        # Init model
        milnet, criterion, optimizer, scheduler = init_model(args)

        fold_best_score = 0
        best_ac = 0
        best_auc = 0
        counter = 0
        best_model = None

        for epoch in range(1, args.num_epochs + 1):
            counter += 1
            train_loss_bag = train(args, train_paths, milnet, criterion, optimizer)
            val_loss_bag, val_avg_score, val_aucs, val_thresholds = test(args, val_paths, milnet, criterion)

            print_epoch_info(epoch, args, train_loss_bag, val_loss_bag, val_avg_score, val_aucs)
            scheduler.step()

            current_score = get_current_score(val_avg_score, val_aucs)
            if current_score > fold_best_score:
                counter = 0
                fold_best_score = current_score
                best_ac = val_avg_score
                best_auc = val_aucs
                best_model = copy.deepcopy(milnet)
                save_model(args, 0, run, save_path, milnet, val_thresholds)

            if counter > args.stop_epochs:
                print("Early stopping triggered.")
                break

        # Final test evaluation
        test_loss_bag, test_avg_score, test_aucs, _ = test(args, test_paths, best_model, criterion)

        # Save summary
        summary_row = {
            "split": 0,
            "test_auc": float(np.mean(test_aucs)),
            "val_auc": float(np.mean(best_auc)),
            "test_acc": float(test_avg_score),
            "val_acc": float(best_ac)
        }

        summary_path = os.path.join(save_path, "single_split_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_row.keys())
            writer.writeheader()
            writer.writerow(summary_row)

        print(f"\nSingle Split Summary:\n"
              f"Test Accuracy: {summary_row['test_acc']:.4f} | Test AUC: {summary_row['test_auc']:.4f}\n"
              f"Val Accuracy: {summary_row['val_acc']:.4f} | Val AUC: {summary_row['val_auc']:.4f}")


    if args.eval_scheme == '5-fold-cv-standalone-test':
        bags_path = glob.glob('temp_train/*.pt')
        bags_path = shuffle(bags_path)
        reserved_testing_bags = bags_path[:int(args.split*len(bags_path))]
        bags_path = bags_path[int(args.split*len(bags_path)):]
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        fold_results = []
        fold_models = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)
            train_path = [bags_path[i] for i in train_index]
            test_path = [bags_path[i] for i in test_index]
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            best_model = []

            for epoch in range(1, args.num_epochs+1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer) # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)
                
                print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(args, fold, run, save_path, milnet, thresholds_optimal)
                    best_model = [copy.deepcopy(milnet.cpu()), thresholds_optimal]
                    milnet.cuda()
                if counter > args.stop_epochs: break
            fold_results.append((best_ac, best_auc))
            fold_models.append(best_model)

        fold_predictions = []
        for item in fold_models:
            best_model = item[0]
            optimal_thresh = item[1]
            test_loss_bag, avg_score, aucs, thresholds_optimal, test_predictions, test_labels = test(args, reserved_testing_bags, best_model.cuda(), criterion, thresholds=optimal_thresh, return_predictions=True)
            fold_predictions.append(test_predictions)
        predictions_stack = np.stack(fold_predictions, axis=0)
        mode_result = mode(predictions_stack, axis=0)
        combined_predictions = mode_result.mode[0]
        combined_predictions = combined_predictions.squeeze()

        if args.num_classes > 1:
            # Compute Hamming Loss
            hammingloss = hamming_loss(test_labels, combined_predictions)
            print("Hamming Loss:", hammingloss)
            # Compute Subset Accuracy
            subset_accuracy = accuracy_score(test_labels, combined_predictions)
            print("Subset Accuracy (Exact Match Ratio):", subset_accuracy)
        else:
            accuracy = accuracy_score(test_labels, combined_predictions)
            print("Accuracy:", accuracy)
            balanced_accuracy = balanced_accuracy_score(test_labels, combined_predictions)
            print("Balanced Accuracy:", balanced_accuracy)

        os.makedirs('test', exist_ok=True)
        with open("test/test_list.json", "w") as file:
            json.dump(reserved_testing_bags, file)

        for i, item in enumerate(fold_models):
            best_model = item[0]
            optimal_thresh = item[1]
            torch.save(best_model.state_dict(), f"test/mil_weights_fold_{i}.pth")
            with open(f"test/mil_threshold_fold_{i}.json", "w") as file:
                optimal_thresh = [float(i) for i in optimal_thresh]
                json.dump(optimal_thresh, file)

    if args.eval_scheme == '5-fold-cv-custom':
        import csv
        import datetime
        import copy

        from sklearn.model_selection import StratifiedKFold, train_test_split

        print('5-fold-cv')
        bags_path = glob.glob('temp_train/*.pt')
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def get_class_label(path, feats_size):
            data = torch.load(path)
            label = data[0, feats_size:].cpu().numpy()
            return int(np.argmax(label))  # assumes one-hot encoding

        labels = [get_class_label(p, args.feats_size) for p in bags_path]

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        summary_rows = []
        fold_results = []

        for fold, (train_index, test_index) in enumerate(kf.split(bags_path, labels)):
            print(f"\n🌀 Starting CV Fold {fold}")
            milnet, criterion, optimizer, scheduler = init_model(args)

            all_train_paths = [bags_path[i] for i in train_index]
            all_train_labels = [labels[i] for i in train_index]

            from collections import Counter

            # Filter out classes with fewer than 2 samples
            label_counts = Counter(all_train_labels)
            valid_classes = [cls for cls, cnt in label_counts.items() if cnt >= 2]

            # Mask to keep only valid samples
            mask = [lbl in valid_classes for lbl in all_train_labels]
            filtered_paths = np.array(all_train_paths)[mask]
            filtered_labels = np.array(all_train_labels)[mask]

            # Stratified split (safe)
            train_path, val_path, train_labels_sub, val_labels_sub = train_test_split(
                filtered_paths,
                filtered_labels,
                test_size=0.3,
                stratify=filtered_labels,
                random_state=fold
            )

            test_path = [bags_path[i] for i in test_index]

            fold_best_score = 0
            best_ac = 0
            best_auc = [0] * args.num_classes
            counter = 0
            best_model = None

            for epoch in range(1, args.num_epochs + 1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer)
                val_loss_bag, val_avg_score, val_aucs, val_thresholds = test(args, val_path, milnet, criterion)

                print_epoch_info(epoch, args, train_loss_bag, val_loss_bag, val_avg_score, val_aucs)
                scheduler.step()

                current_score = get_current_score(val_avg_score, val_aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = val_avg_score
                    best_auc = val_aucs
                    best_model = copy.deepcopy(milnet)
                    save_model(args, fold, run, save_path, milnet, val_thresholds)
                if counter > args.stop_epochs:
                    break

            # Final test on held-out test set using best model
            test_loss_bag, test_avg_score, test_aucs, _ = test(args, test_path, best_model, criterion)

            summary_rows.append({
                "folds": fold,
                "test_auc": float(np.mean(test_aucs)),
                "val_auc": float(np.mean(best_auc)),
                "test_acc": float(test_avg_score),
                "val_acc": float(best_ac)
            })
            fold_results.append((test_avg_score, test_aucs))

            print(f"✅ Fold {fold} completed | Test Acc: {test_avg_score:.4f} | Test AUCs: {[round(a, 3) for a in test_aucs]}")

        # Save results summary CSV
        csv_path = os.path.join(save_path, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["folds", "test_auc", "val_auc", "test_acc", "val_acc"])
            writer.writeheader()
            writer.writerows(summary_rows)

        # Print final aggregated metrics
        mean_ac = np.mean([r["test_acc"] for r in summary_rows])
        mean_auc = np.mean([r["test_auc"] for r in summary_rows])
        print(f"\n📊 Final Summary:\nMean Test Accuracy: {mean_ac:.4f} | Mean Test AUC: {mean_auc:.4f}")

                

if __name__ == '__main__':
    main()
EOF


echo "🎯 Training DSMIL model on extracted features..."
MPLBACKEND=Agg conda run -n dsmil python train_tcga.py \
    --dataset=ndpi_files \
    --num_classes=3 \
    --feats_size=1024 \
    --num_epochs=300 \
    --dropout_patch=0.60 \
    --lr=1e-4 \
    --weight_decay=1e-4 \
    --stop_epochs=100 \
    --eval_scheme=5-fold-cv-custom \
    --split=0.2



echo "✅ All steps completed successfully."
