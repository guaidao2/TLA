"""训练脚本：变速率世界模型 + 误差驱动学习（无 BP），打印学习曲线与判据。

支持直接运行（python examples/train_toy.py）与模块运行（python -m examples.train_toy）。
"""
import os
import sys

# 路径引导：无论从哪运行，都把仓库根加入 sys.path（否则直接按路径运行找不到 tla 包）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tla.criteria.run_criteria import run_criteria

if __name__ == "__main__":
    r = run_criteria(verbose=True)
    print(f"\n训练 MSE(末段)={r['train_mse']:.4f}  未见速度测试 MSE={r['mse_test']:.4f}")
    print(f"P-LEARN-3={'PASS' if r['p_learn3'] else 'FAIL'}  P-COG-1={'PASS' if r['p_cog1'] else 'FAIL'}")
