# PEAF: Payload Entropy-based Adversarial Framework

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation of the *Payload Entropy-based Adversarial Framework (PEAF)* for generating adversarial attacks against network intrusion detection systems (NIDS), based on the research paper "Adversarial Attacks on Network Traffic Detection Using Payload Entropy Features and Universal Perturbations".

## 📋 Overview

PEAF is a novel adversarial attack framework that exploits entropy-based features in network traffic to generate sophisticated adversarial examples that can evade AI-powered intrusion detection systems while maintaining complete traffic functionality.

### Key Features

- *Dual-Module Architecture*: 
  - *Low Entropy Module (LEM)*: Targets structured traffic (Botnet, Brute Force, PortScan, Web Attacks)
  - *High Entropy Module (HEM)*: Targets high-randomness traffic (DDoS attacks)
- *Universal Perturbations*: Generates transferable adversarial examples across different model architectures
- *Entropy-Aware Attacks*: Exploits statistical characteristics of network traffic payloads
- *Gradient-Based Optimization*: Uses CNN loss functions for iterative perturbation refinement
- *Traffic Functionality Preservation*: Maintains original attack effectiveness while evading detection

## 🎯 Attack Effectiveness

Our implementation achieves significant performance degradation on neural network-based NIDS:

| Model | Baseline Accuracy | After PEAF Attack | Reduction |
|-------|------------------|-------------------|-----------|
| 1D-CNN | 97.16% | 24.40% | *72.76%* |
| SDAE | 92.12% | 21.23% | *78.77%* |

## 🛠 Installation

### Requirements

bash
Python >= 3.7
TensorFlow >= 2.0
NumPy >= 1.19.0
Pandas >= 1.1.0
Scikit-learn >= 0.24.0
Matplotlib >= 3.3.0
Seaborn >= 0.11.0
SciPy >= 1.5.0


### Quick Install

bash
git clone https://github.com/mehedi93hasan/Payload-Entropy-based-Adversarial-Framework.git

cd Payload-Entropy-based-Adversarial-Framework

## 📊 Dataset

This implementation uses the *CIC-IDS2017* dataset, a comprehensive network intrusion detection dataset containing:

- *Botnet*: Bot traffic patterns
- *Brute Force*: FTP/SSH brute force attacks  
- *PortScan*: Network port scanning activities
- *Web Attack*: SQL injection, XSS, brute force web attacks
- *DDoS*: Distributed denial of service attacks

### Data Preprocessing

The framework automatically handles:
- Dataset balancing according to paper specifications
- Entropy feature extraction from flow-level statistics
- Label consolidation and mapping
- Train/validation/test splits

# Load your CIC-IDS2017 dataset
df = pd.read_csv('your_cic_ids2017_data.csv')

# Initialize and train models
cnn_model = CNNModel(input_shape=(6,))  # 6 entropy features
cnn_model.compile_model()

# Generate PEAF attacks
peaf = PEAFAttack(cnn_model.model)
adversarial_samples = peaf.generate_adversarial_samples(
    X_test, y_test, 
    num_perturbations=5,
    iterations=100
)

# Evaluate attack effectiveness
baseline_acc = cnn_model.model.evaluate(X_test, y_test)[1]
adversarial_acc = cnn_model.model.evaluate(adversarial_samples, y_test)[1]
reduction = ((baseline_acc - adversarial_acc) / baseline_acc) * 100

print(f"Accuracy reduction: {reduction:.2f}%")


### Complete Example

python
# Run the complete PEAF attack pipeline
python peaf_attack_demo.py --dataset path/to/cic_ids2017.csv --output results/


## 📁 Project Structure


peaf-attack/
├── README.md
├── requirements.txt
├── LICENSE
├── peaf_attack/
│   ├── __init__.py
│   ├── models.py              # CNN and SDAE model implementations
│   ├── attack.py              # PEAF attack algorithm
│   ├── feature_extraction.py  # Entropy feature engineering
│   ├── data_preprocessing.py  # Dataset balancing and preprocessing
│   └── utils.py              # Utility functions
├── notebooks/
│   └── PEAF_Attack_Demo.ipynb # Google Colab notebook
├── examples/
│   └── peaf_attack_demo.py   # Complete example script
├── results/
│   └── sample_results.json   # Sample attack results
└── docs/
    ├── architecture.md       # Detailed architecture explanation
    └── methodology.md        # Attack methodology details


## 🔬 Methodology

### 1. Traffic Characterization
- Analysis of payload-level statistical characteristics
- Entropy pattern identification across attack categories
- Feature engineering for entropy-based analysis

### 2. Dual-Module Attack Architecture
- *LEM*: Increases entropy in structured traffic through controlled randomness
- *HEM*: Reduces entropy in high-randomness traffic through pattern introduction

### 3. Universal Perturbation Generation
- Gradient-based optimization targeting entropy-sensitive features
- Iterative refinement with traffic functionality constraints
- Cross-model transferability ensuring black-box effectiveness

### 4. Entropy Masking Mechanism
- Strategic perturbation injection at maximum gradient sensitivity points
- Payload segment analysis for optimal modification locations
- Statistical disruption while preserving protocol compliance

## 📈 Results Analysis

### White-box Attack Performance (1D-CNN)

| Traffic Type | Baseline Acc. | Post-Attack Acc. | Reduction | Module |
|--------------|---------------|------------------|-----------|---------|
| Botnet | 98.60% | 26.12% | 73.54% | LEM |
| Brute Force | 99.17% | 22.83% | 77.01% | LEM |
| PortScan | 96.70% | 28.91% | 70.12% | LEM |
| Web Attack | 94.74% | 31.57% | 66.69% | LEM |
| DDoS | 96.59% | 23.64% | 75.52% | HEM |

### Black-box Transferability (SDAE)

| Traffic Type | Baseline Acc. | PEAF Attack | Reduction |
|--------------|---------------|-------------|-----------|
| Botnet | 89.51% | 30.46% | 65.95% |
| Brute Force | 91.67% | 28.12% | 69.33% |
| PortScan | 88.79% | 25.91% | 70.81% |
| Web Attack | 100.00% | 35.71% | 64.29% |
| DDoS | 93.71% | 19.23% | 79.47% |

## ⚙ Configuration

### Model Parameters

python
# CNN Model Configuration
CNN_CONFIG = {
    'filters': [64, 128, 256],
    'kernel_size': 3,
    'dropout_rate': 0.3,
    'learning_rate': 1e-6,
    'batch_size': 32,
    'epochs': 1000
}

# PEAF Attack Configuration
PEAF_CONFIG = {
    'entropy_threshold': 4.0,
    'learning_rate': 0.01,
    'iterations': 100,
    'perturbation_counts': [1, 3, 5, 7],
    'noise_factor': 0.1,
    'smoothing_factor': 0.1
}


### Entropy Feature Configuration

python
ENTROPY_FEATURES = {
    'PayloadEntropy': 'Shannon entropy of payload content',
    'SizeEntropy': 'Entropy based on packet sizes',
    'HeaderComplexity': 'Header structure complexity',
    'CompressionRatio': 'Data compression characteristics',
    'RandomnessIndex': 'Traffic randomness measure',
    'InformationDensity': 'Information content density'
}

# Test specific attack configurations
from peaf_attack import ExperimentRunner

runner = ExperimentRunner(config_path='config/custom_config.yaml')
results = runner.run_experiment(
    perturbation_range=[1, 5, 10],
    traffic_types=['botnet', 'ddos'],
    models=['cnn', 'sdae']
)


## 📊 Visualization

Generate comprehensive attack analysis visualizations:

python
from peaf_attack.visualization import AttackVisualizer

visualizer = AttackVisualizer()
visualizer.plot_attack_effectiveness(results)
visualizer.plot_entropy_analysis(entropy_features)
visualizer.plot_module_comparison(lem_results, hem_results)
visualizer.generate_report(output_path='reports/attack_analysis.html')


## 🔒 Ethical Considerations

This research is intended for:
- ✅ *Academic research* and cybersecurity education
- ✅ *Defensive security* system evaluation and improvement
- ✅ *Vulnerability assessment* of existing NIDS
- ✅ *Robustness testing* of AI-based security systems

*⚠ Important*: This framework should only be used for legitimate security research and authorized penetration testing. Users are responsible for ensuring compliance with applicable laws and regulations.




## 🙏 Acknowledgments

- Original research paper authors
- CIC-IDS2017 dataset creators
- TensorFlow and scikit-learn communities
- Cybersecurity research community

## 📈 Roadmap

- [ ] Real PCAP payload entropy implementation
- [ ] Additional model architectures (Transformer, LSTM)
- [ ] Defense mechanism evaluation
- [ ] Real-time attack detection
- [ ] Extended dataset support (UNSW-NB15, NSL-KDD)
- [ ] GUI interface for attack visualization

---

⭐ *Star this repository* if you find it useful for your research!

🔔 *Watch* for updates and new features! Share PEAF Attack GitHub READM
