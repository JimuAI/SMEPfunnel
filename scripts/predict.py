import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import joblib
import torch
from torch.utils.data import DataLoader
import pandas as pd
from Bio import SeqIO
from scripts.data_processing import DataGenerator
from datasets import PredictDataset


def xgboost_classifier_predict(X_pred):
    # 获取模型
    model = joblib.load("models/xgboost_classifier_model.pkl")
    # 预测
    y_pred = model.predict(X_pred)
    return y_pred
    
def xgboost_regressor_predict(X_pred):
    # 获取模型
    model = joblib.load("models/xgboost_regressor_model.pkl")
    # 预测
    y_pred = model.predict(X_pred)
    return y_pred

def lstm_regressor_predict(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # 获取模型
    model = torch.load("models/lstm_best_model.pth", weights_only=False).to(device)
    model.eval()
    # 批量预测
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            y_pred = model(batch)
            all_preds.append(y_pred.cpu())
    return torch.cat(all_preds, dim=0).numpy()


def predict_workflow(input_file):
    # 读取数据
    suffix = input_file.split(".")[-1]
    if suffix == "txt":
        with open(input_file, "r", encoding="utf-8") as f:
            sequences = [line.strip() for line in f]
    elif suffix == "fasta":
        sequences = [str(record.seq) for record in SeqIO.parse(input_file, "fasta")]
    else:
        raise ValueError("Unsupported file format. Please use txt or fasta.")
    # 处理数据
    ordered_results = DataGenerator.gen_peptide_descriptor_parallel(sequences, num_workers=4)
    feature_df = pd.concat([pd.DataFrame({"sequence": sequences}), pd.DataFrame(ordered_results)], axis=1) 
    
    # 分类预测
    feature_X = feature_df.iloc[:, 1:]  # 特征数据
    y_pred = xgboost_classifier_predict(feature_X)
    pred_pos_feature_df = feature_df[y_pred == 1].reset_index(drop=True)
    
    # xgboost 回归预测
    feature_X = pred_pos_feature_df.iloc[:, 1:]
    xgboost_pred = xgboost_regressor_predict(feature_X)
    
    # lstm 回归预测
    sequences = pred_pos_feature_df["sequence"].to_list()
    dataset = PredictDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    lstm_pred = lstm_regressor_predict(dataloader)
    
    # 3. 返回结果
    results_df = pd.DataFrame({
        "sequence": pred_pos_feature_df["sequence"].values,  # 序列
        "xgboost_pred": xgboost_pred,  # XGBoost 预测值
        "lstm_pred": lstm_pred  # LSTM 预测值
    })
    results_df.to_csv("data/predict/prediction_results.csv", index=False)  # 不保存索引
    
if __name__ == '__main__':
    predict_workflow("data/predict/sequences.txt")
    