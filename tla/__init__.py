"""TLA —— 公理组合式神经架构（Axiom-Composed Neural Architecture）。

v0.2 组合体最小实现：LTC 基板 + 无 BP 误差驱动学习 + 自适应推理深度 + Self_Slot + 生长/修剪 + CLS 重放。

铁律：主网络学习环只用局部误差（无 autograd/BP）；管理性子网（Self_Slot 除外）可正常训练。
"""
__version__ = "0.2.0"
