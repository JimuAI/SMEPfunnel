# SMEP funnel


**SMEP (Sequential Model Ensemble Pipeline) funnel** is a machine learning-based workflow for **antimicrobial peptide (AMP) screening**. It integrates an **XGBoost classifier**, an **XGBoost regressor**, and an **LSTM regressor** to sequentially filter and predict AMP activity.


> **Note:** This repository provides a machine learning pipeline specifically designed for predicting antimicrobial peptide (AMP) activity against *Staphylococcus aureus* (S. aureus). It is tested and recommended to run with **Python 3.10**.


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/JimuAI/SMEPfunnel.git
cd SMEPfunnel
pip install -r requirements.txt
```



## Usage

Prepare input peptide sequences in a text file and run the pipeline for predicting:

```bash
python scripts/predict.py data/predict/sequences.txt
```



## Reference

```latex
@article{huang2023identification,
  title={Identification of potent antimicrobial peptides via a machine-learning pipeline that mines the entire space of peptide sequences},
  author={Huang, Junjie and Xu, Yanchao and Xue, Yunfan and Huang, Yue and Li, Xu and Chen, Xiaohui and Xu, Yao and Zhang, Dongxiang and Zhang, Peng and Zhao, Junbo and others},
  journal={Nature Biomedical Engineering},
  volume={7},
  number={6},
  pages={797--810},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
