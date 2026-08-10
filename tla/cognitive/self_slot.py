"""Self_Slot（⑦ 自模型）：从"自身输入状态"预测"自己会输出什么"。

- self_pred = Self_Slot(x_t)，target = p_out_t（当前自己的输出，detached）；
- x_t = [obs_t, h_t]（完整输入，不含输出本身——避免"复制输入"的平凡解）；
- 这是网络的"输入→输出"自模型（预测我自己，不预测世界）：训练分布上学习输入-输出映射；
- 推理一致性门：自模型对**训练分布外**输入的分歧大 → 自身状态异常 → 继续琢磨（inference.py）；
- 固定随机特征 + 末层局部误差驱动更新（ΔW = η·e·φᵀ）——保持无 BP 纪律。
"""
import torch


class SelfSlot:
    def __init__(self, in_dim, feat_dim=64, out_dim=2, lr=0.005, seed=None):
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.F = torch.randn(in_dim, feat_dim, generator=gen) / (in_dim ** 0.5)
        self.W = torch.randn(out_dim, feat_dim, generator=gen) / (feat_dim ** 0.5)
        self.b = torch.zeros(out_dim)
        self.lr = lr

    def _phi(self, state):
        return torch.tanh(self.F.T @ state)

    def predict(self, state):
        return self.W @ self._phi(state) + self.b

    def learn(self, state, target):
        phi = self._phi(state)
        p = self.W @ phi + self.b
        e = target - p
        self.W = self.W + self.lr * torch.outer(e, phi)
        self.b = self.b + self.lr * e
        return float(torch.mean(e ** 2).item())
