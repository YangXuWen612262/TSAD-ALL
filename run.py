import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.OracleAD import OracleAD
from exp.Exp_OracleAD import train_oraclead, OracleADLossConfig

def set_seed(seed: int):
    """固定随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 保证卷积等操作的确定性 (会稍微牺牲性能)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description="OracleAD Training & Evaluation")

    # --- 数据参数 ---
    parser.add_argument("--num_vars", type=int, default=5, help="变量数量 (N)")
    parser.add_argument("--window_len", type=int, default=10, help="时间窗口长度 (L)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--num_samples", type=int, default=1000, help="生成的虚拟样本数量")

    # --- 模型架构参数 ---
    parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏层维度 (H)")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM 层数")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention 头数 (需被 H 整除)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")

    # --- 训练参数 ---
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="训练设备")

    # --- Loss 参数 (OracleAD 特有) ---
    parser.add_argument("--lambda_recon", type=float, default=0.1, help="重构 Loss 权重")
    parser.add_argument("--lambda_dev", type=float, default=3.0, help="偏差 (Deviation) Loss 权重")
    parser.add_argument("--sls_epoch", type=int, default=2, help="从第几轮开始应用 SLS 约束")

    args = parser.parse_args()
    return args

def main():
    args=get_args()

    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Configurations: {vars(args)}")
    print(f"Running on: {device}")

    #TODO: 加载数据集并进行划分得到train_loader test_loader 

    # 这里先跳过数据逻辑，默认已经得到数据

    model=OracleAD(
        num_vars=args.num_vars,
        window_len=args.window_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
    )

    Loss_config=OracleADLossConfig(
        lambda_recon=args.lambda_recon,
        lambda_dev=args.lambda_dev,
        sls_epoch=args.sls_epoch,
    )

    

