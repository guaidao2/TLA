"""TLAPR1MoEModel：摊销首猜 × MoE 专家分离 组合体装配。

复用 TLAPR1Model 的训练/推理/回退/重放循环，仅换装 AmortizedMoEPCN 堆栈
（原则一的首猜承重 + 专家分离残差）——接口完全对齐，行为差异全在堆栈内。
"""
from tla.model_pr1 import TLAPR1Model
from tla.cognitive.pcn_amortized_moe import AmortizedMoEPCN


class TLAPR1MoEModel(TLAPR1Model):
    def __init__(self, obs_dim=3, out_dim=2, ltc_hidden=16, hidden=24, top=16,
                 n_experts=2, **kw):
        kw["stack_cls"] = AmortizedMoEPCN
        kw.setdefault("n_experts", n_experts)
        kw.setdefault("top", top)
        super().__init__(obs_dim=obs_dim, out_dim=out_dim, ltc_hidden=ltc_hidden,
                         hidden=hidden, **kw)
