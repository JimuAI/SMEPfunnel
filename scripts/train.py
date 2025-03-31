import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import xgboost as xgb
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from scripts.datasets import load_class_feature_data, load_regress_feature_data, load_regress_seq_data
from models.xgboost import xgboost_classifier, xgboost_regressor
from models.lstm import LSTMRegressor
from torch.utils.tensorboard import SummaryWriter
import joblib

def train_xgboost_classifier():
    # 加载数据
    file_path = "data/processed/all_feature.csv"
    X_train, X_test, y_train, y_test = load_class_feature_data(file_path)
    # 初始化 XGBoost 分类器
    model = xgboost_classifier()
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估模型
    acc = accuracy_score(y_test, y_pred)
    print(f"acc: {acc:.4f}")
    # 保存模型
    joblib.dump(model, "models/xgboost_classifier_model.pkl")
    


def train_xgboost_regressor():
    # 加载数据
    file_path = "data/processed/all_feature.csv"
    X_train, X_test, y_train, y_test = load_regress_feature_data(file_path, log_base=10)
    # 初始化 XGBoost 回归器
    model = xgboost_regressor()
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"测试集 MAE: {mae:.4f}")
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 R2: {r2:.4f}")
    
    # 为防止过拟合，使用训练集评估模型
    print(f"训练集 MSE: {mean_squared_error(y_train, model.predict(X_train)):.4f}")
    print(f"训练集 MAE: {mean_absolute_error(y_train, model.predict(X_train)):.4f}")
    print(f"训练集 R2: {r2_score(y_train, model.predict(X_train)):.4f}")
    
    # 保存模型
    joblib.dump(model, "models/xgboost_regressor_model.pkl")

def train_lstm_regressor(data_path, 
                         batch_size, max_len,
                         embedding_dim, hidden_num, num_layer, bidirectional, dropout,
                         epochs, lr,
                         log_dir, model_save_path):
    # 加载数据
    train_loader, test_loader = load_regress_seq_data(data_path, batch_size, max_len)
    # 自动检测设备（CUDA / MPS / CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # 初始化 LSTM 回归器
    model = LSTMRegressor(input_dim=21, embedding_dim=embedding_dim, hidden_dim=hidden_num, num_layers=num_layer,
                          bidirectional=bidirectional, dropout=dropout).to(device)
    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    # TensorBoard 记录器
    writer = SummaryWriter(log_dir=log_dir)
    # 确保模型保存目录存在
    model_save_path = Path(model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    # 记录最佳 loss
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        start_time = time.time()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False) as pbar:
            for sequences, labels in pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")  # 更新进度条显示 loss
        avg_train_loss = train_loss / len(train_loader)
        
        # 测试模型
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        
        # 记录 loss 到 TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Test", avg_test_loss, epoch + 1)
        
        # 保存最佳模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_path = os.path.join(model_save_path, f"best_model_step{epoch}.pth")
            torch.save(model, best_model_path)
            print(f"新的最佳模型已保存！ (epoch={epoch+1}, test_loss={best_loss:.4f})")
        
        epoch_time = time.time() - start_time
        # 打印日志
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Time: {epoch_time:.2f}s")
    # 关闭 TensorBoard 记录器
    writer.close()


if __name__ == "__main__":
    # 训练 XGBoost 分类器
    # train_xgboost_classifier()
    
    # 训练 XGBoost 回归器
    # train_xgboost_regressor()
    
    # 训练 LSTM 回归器
    train_lstm_regressor(
        batch_size=32,
        max_len=50,
        embedding_dim=16,
        hidden_num=64,
        num_layer=2,
        bidirectional=True,
        dropout=0.5,
        epochs=500,
        lr=0.001,
        data_path="data/processed/all_feature.csv",
        log_dir="logs/lstm_regressor_all",
        model_save_path="models/lstm_regressor_all"
    )