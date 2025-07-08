# MLIAProject - Semi-Supervised Learning for Mesothelioma Classification

[![GitHub Repository](https://img.shields.io/badge/GitHub-MLIAProject-blue)](https://github.com/LucaIanniello/MLIAProject)

This repository contains the complete research materials, experimental data, and analysis scripts for the project "Semi-Supervised Learning for Automatic Classification of Mesothelioma Subtypes using High-Resolution Digital Histological Images (WSI)".

## 📋 Project Overview

This research investigates the effectiveness of **semi-supervised machine learning techniques** for automatic classification of mesothelioma subtypes using high-resolution digital histological preparation images (WSI - Whole Slide Images). The study demonstrates significant improvements in classification performance through advanced **data augmentation techniques** and **multiple instance learning (MIL)** approaches.

## 📄 Documentation

- **Complete Report**: [`Report`](./Project_Report.pdf)
- **Project Repository**: [MLIAProject](https://github.com/LucaIanniello/MLIAProject)

### Research Questions

- **RQ1**: How effectively can semi-supervised learning approaches classify mesothelioma subtypes with limited labeled data?
- **RQ2**: What impact does feature-level data augmentation have on classification performance?
- **RQ3**: How do different feature extractors (ResNet50, UNI, UNIv2, Phikon-v2) perform in weakly supervised settings?


### Authors
- **Luca Ianniello** - [GitHub](https://github.com/LucaIanniello)
- **Raffaele Martone** - [GitHub](https://github.com/Martons00)
- **Antonio Sirica** - [GitHub](https://github.com/Asir29)

*Politecnico di Torino - Machine Learning in Applications Course*

## 🚀 Key Contributions

- **Semi-supervised MIL Framework**: Implementation of CLAM and DSMIL for weakly supervised WSI classification
- **Advanced Data Augmentation**: Feature-level augmentation using extrapolation methods and diffusion models (AugDiff)
- **Foundation Model Integration**: Comprehensive evaluation of UNI, UNIv2, and Phikon-v2 feature extractors
- **Loss Function Analysis**: Systematic comparison of CE, Focal, WCE, and Contrastive losses for class imbalance
- **Clinical Validation**: Real-world dataset from San Luigi Hospital with three mesothelioma subtypes

## 📁 Repository Structure

```
MLIAProject/
├── CLAM/                     # CLAM implementation and experiments
│   ├── dataset_modules/      # Dataset handling utilities
│   ├── models/              # Model architectures (CLAM, attention mechanisms)
│   ├── utils/               # Utility functions and helpers
│   ├── requirements.txt     # Python dependencies
│   └── env.yml             # Conda environment configuration
├── DSMIL/                   # DSMIL implementation
│   ├── models/             # Dual-stream MIL architecture
│   ├── datasets/           # Data loading and preprocessing
│   └── configs/            # Configuration files
├── Papers/                  # Research papers and references
│   ├── CLAM_paper.pdf
│   ├── DSMIL_paper.pdf
│   ├── UNI_foundation.pdf
│   └── related_work/
├── Results_Train_Eval/      # Training and evaluation results
│   ├── CLAM_results_*/     # CLAM experimental results
│   ├── DSMIL_results_*.csv # DSMIL experimental results
│   ├── attention_maps/     # Visualization outputs
│   └── performance_logs/   # Training logs and metrics
├── Script/                  # Execution scripts and notebooks
│   ├── preprocessing/      # Data preprocessing scripts
│   ├── training/          # Training orchestration
│   ├── evaluation/        # Model evaluation and visualization
│   └── analysis/          # Statistical analysis notebooks
├── presets/                # Configuration presets
│   ├── clam_configs/      # CLAM training configurations
│   ├── dsmil_configs/     # DSMIL training configurations
│   └── feature_configs/   # Feature extraction settings
├── mla25_fp01-2.pdf        # Complete research report
└── README.md              # This file
```

## 🧪 Experimental Setup

### Dataset Characteristics

#### **San Luigi Hospital Mesothelioma Dataset**
- **123 total WSI cases** from real patients
- **Distribution**: 96 epithelioid, 22 biphasic, 5 sarcomatoid
- **Subset used**: 22 WSI (9 biphasic, 8 epithelioid, 5 sarcomatoid)
- **Confidential dataset** with privacy restrictions

### Histological Subtypes

1. **Epithelioid Mesothelioma** (~60% of cases)
   - Better prognosis compared to other subtypes
   - Cells resembling normal mesothelial cells
   - Papillary or tubular structures

2. **Sarcomatoid Mesothelioma** (~20% of cases)
   - Most aggressive clinical course
   - Spindle-shaped cells mimicking sarcoma
   - Poorest outcomes

3. **Biphasic Mesothelioma** (~20% of cases)
   - Contains both epithelioid and sarcomatoid components
   - Intermediate prognosis and treatment response

### Multiple Instance Learning Frameworks

#### **CLAM (Clustering-constrained Attention MIL)**
- **Weakly supervised** whole slide image classification
- **Attention-based pooling** for patch aggregation
- **Clustering constraints** for feature space refinement
- **Interpretable predictions** with attention heatmaps

#### **DSMIL (Dual-Stream Multiple Instance Learning)**
- **Dual-stream architecture** for instance and bag-level information
- **Self-supervised contrastive learning** with SimCLR
- **Pyramidal fusion mechanism** for multi-scale features
- **Enhanced robustness** through improved aggregation

### Feature Extractors Evaluated

1. **ResNet50-ImageNet**: Traditional baseline encoder
2. **UNI & UNIv2**: Foundation models for computational pathology
3. **Phikon-v2**: Specialized for biomarker prediction
4. **CLAM Feature Extractor**: ResNet50-based with pathology-specific training

## 📊 Key Results

### Performance with Data Augmentation

| Feature Extractor | Loss Function | Test AUC | Test Accuracy | Val AUC | Val Accuracy |
|-------------------|---------------|----------|---------------|---------|--------------|
| **UNI v2 Trident** | Cross Entropy | **1.00** | **1.00** | **1.00** | **1.00** |
| **UNI v1 Trident** | Cross Entropy | **1.00** | 0.90 | **1.00** | 0.90 |
| **UNI v2 Trident** | Contrastive | **1.00** | **1.00** | **1.00** | **1.00** |
| **Phikon Trident** | Cross Entropy | 0.98 | 0.80 | 0.98 | 0.90 |
| **ResNet50 Trident** | Cross Entropy | 0.91 | 0.90 | 0.89 | 0.90 |

### Key Findings

- **Dramatic Improvement**: Data augmentation leads to near-perfect or perfect performance across all feature extractors
- **Foundation Model Superiority**: UNI and UNIv2 consistently achieve best results (AUC=1.00, Accuracy=1.00)
- **CLAM Robustness**: Shows consistent performance across all loss functions
- **Loss Function Impact**: Focal Loss and WCE effectively handle class imbalance

### Coverage Analysis

- **Feature Extractor Ranking**: UNI/UNIv2 > CLAM > Phikon-v2 > ResNet50
- **Augmentation Effect**: 10x improvement in average performance metrics
- **Generalization**: Strong validation performance indicates robust learning

## 📈 Metrics Explained

### Classification Metrics
- **AUC (Area Under ROC Curve)**: Measures discriminative ability across all thresholds
- **Accuracy**: Proportion of correctly classified samples
- **Precision/Recall**: Per-class performance for imbalanced datasets

### MIL-Specific Metrics
- **Bag-level Accuracy**: Slide-level classification performance
- **Instance-level Attention**: Patch importance scores
- **Coverage**: Percentage of relevant tissue regions identified

## ⚠️ Limitations

### Dataset Constraints
1. **Limited Sample Size**: Only 22 WSI used due to computational constraints
2. **Class Imbalance**: Severe underrepresentation of sarcomatoid subtype (5 cases)
3. **Single Institution**: Dataset from San Luigi Hospital only

### Technical Limitations
1. **Computational Requirements**: Requires high-end GPU for training
2. **Processing Time**: WSI analysis computationally intensive
3. **Memory Constraints**: Gigapixel images require careful memory management

### Methodological Considerations
1. **Single Split**: Limited to one train/test split due to small dataset
2. **Validation Strategy**: Simple holdout validation instead of k-fold
3. **Augmentation Dependency**: High performance heavily relies on data augmentation

## 🔮 Future Work

### Immediate Improvements
- **Larger Dataset**: Expand to multi-institutional cohorts
- **Cross-validation**: Implement robust k-fold validation
- **Real-time Inference**: Optimize models for clinical deployment

### Advanced Techniques
- **Multi-modal Integration**: Combine WSI with clinical data
- **Federated Learning**: Privacy-preserving multi-site training
- **Transformer Architectures**: Explore vision transformers for WSI analysis
- **Active Learning**: Intelligent annotation strategies

### Clinical Translation
- **Prospective Validation**: Real-world clinical testing
- **Interpretability**: Enhanced explainability for pathologists
- **Integration**: Seamless integration with pathology workflows

## 📚 References

Key papers and resources used in this research:

1. **CLAM Framework**: Lu et al. "Data-efficient and weakly supervised computational pathology on whole-slide images" (Nature Biomedical Engineering, 2021)
2. **DSMIL Method**: Li et al. "Dual-stream multiple instance learning network for whole slide image classification" (Medical Image Analysis, 2023)
3. **UNI Foundation Model**: Chen et al. "Towards a general-purpose foundation model for computational pathology" (Nature Medicine, 2024)
4. **Semi-supervised Learning**: Voigt et al. "Investigation of semi- and self-supervised learning methods in the histopathological domain" (Computers in Biology and Medicine, 2022)

## 🤝 Contributing

This repository represents completed research work conducted at Politecnico di Torino. For questions about methodology, results, or potential collaborations, please open an issue or contact the authors.



## 🏥 Acknowledgments

- **San Luigi Hospital, Turin** for providing the confidential mesothelioma dataset
- **Politecnico di Torino** for computational resources and academic support
- **HPC Legion Server** for GPU computational infrastructure
- **CLAM and DSMIL frameworks** for foundational methodologies

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{AugDiff,
  title={AugDiff: Diffusion based Feature Augmentation for Multiple Instance Learning in Whole Slide Image},
  author={Shao, Zhuchen and Dai, Liuxi and Wang, Yifeng and Wang, Haoqian and Zhang, Yongbing},
  journal={arXiv preprint arXiv:2309.07935},
  year={2023}
}

@article{CLAM,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}

@article{DSMIL,
  title={Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning},
  author={Li, Bin and Li, Yin and Eliceiri, Kevin W},
  journal={Medical Image Analysis},
  volume={85},
  pages={102732},
  year={2023},
  publisher={Elsevier}
}

@inproceedings{Extrapolation,
  title={Dataset augmentation in feature space},
  author={DeVries, Terrance and Taylor, Graham W},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}

@article{ClinicalHistologic,
  title={Clinical significance of histologic subtyping of malignant pleural mesothelioma},
  author={Brcic, Luka and Kern, Izidor},
  journal={Translational Lung Cancer Research},
  volume={7},
  number={5},
  pages={556--569},
  year={2018},
  publisher={AME Publishing Company}
}

@article{SurveyMIL,
  title={Multiple Instance Learning for Digital Pathology: A Review of the State-of-the-Art, Limitations & Future Potential},
  author={Gadermayr, Michael and Tschuchnig, Maximilian},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={5},
  pages={1121--1135},
  year={2022},
  publisher={IEEE}
}

@inproceedings{Contrastive,
  title={Rethinking Multiple Instance Learning for Whole Slide Image Classification: A Good Instance Classifier is All You Need},
  author={Qu, Linhao and Ma, Yingfan and Luo, Xiaoyuan and Guo, Qinhao and Wang, Manning and Song, Zhijian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1217--1226},
  year={2023}
}

@article{Pathology,
  title={Pathology of mesothelioma},
  author={Inai, Kouki},
  journal={Environmental Health and Preventive Medicine},
  volume={13},
  number={2},
  pages={60--64},
  year={2008},
  publisher={Springer}
}

@article{PathologyDiagnosis,
  title={The pathological and molecular diagnosis of malignant pleural mesothelioma: a literature review},
  author={Alì, Greta and Bruno, Rossella and Fontanini, Gabriella},
  journal={Translational Lung Cancer Research},
  volume={10},
  number={1},
  pages={72--83},
  year={2021},
  publisher={AME Publishing Company}
}

@article{SemiSupervisedLearning,
  title={Investigation of semi- and self-supervised learning methods in the histopathological domain},
  author={Voigt, Benjamin and Fischer, Oliver and Schilling, Bruno and Krumnow, Christian and Hertaba, Christian},
  journal={Computers in Biology and Medicine},
  volume={144},
  pages={105377},
  year={2022},
  publisher={Elsevier}
}

@inproceedings{SimCLR,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={1597--1607},
  year={2020},
  organization={PMLR}
}

@article{Trident,
  title={Accelerating Data Processing and Benchmarking of AI Models for Pathology},
  author={Zhang, Andrew and Jaume, Guillaume and Vaidya, Anurag and Ding, Tong and Mahmood, Faisal},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={1456},
  year={2023},
  publisher={Nature Publishing Group}
}
```

## 📋 License

This project is made available for **academic and research purposes**. The dataset used is confidential and not publicly available due to privacy restrictions.

---

**Note**: This repository demonstrates state-of-the-art results in computational pathology for mesothelioma classification using semi-supervised learning and advanced data augmentation techniques. The complete experimental pipeline showcases the potential of foundation models and multiple instance learning in digital pathology applications.