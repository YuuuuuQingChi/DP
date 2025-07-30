# 创建 diagnosis.py 文件
import sys
import torch

print("="*50)
print(f"Python 可执行文件路径: {sys.executable}")
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 路径: {torch.__file__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print("="*50)