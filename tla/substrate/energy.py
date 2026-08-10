"""Energy_Budget 疲劳抑制（⑪ 能量-时间-精度）+ 断电语义（② 物理崩=全崩）。

- 每个推理步消耗能量（激活越多越耗）；
- 能量耗尽 → 断电 → 上层输出被抑制（suppressed 标记）；
- 软衰减 + 滞回（fatigue_threshold 以下先抑制低 importance 激活，见 meta/importance.py 配合）。
"""
from dataclasses import dataclass


@dataclass
class EnergyState:
    level: float
    temperature: float  # 1 - integrity（模拟发热）
    integrity: float    # level / capacity（完整性）


class EnergyBudget:
    def __init__(self, capacity=10.0, fatigue_threshold=0.2, base_cost=1.0):
        self.capacity = float(capacity)
        self.level = float(capacity)
        self.fatigue_threshold = float(fatigue_threshold)
        self.base_cost = float(base_cost)

    def reset(self):
        self.level = self.capacity

    def step_cost(self, n_active=1.0):
        """激活越多越耗能（⑪）。"""
        return self.base_cost * (0.5 + 0.5 * n_active)

    def consume(self, n_active=1.0):
        """消耗一步能量。返回 False 表示已断电。"""
        self.level -= self.step_cost(n_active)
        if self.level <= 0.0:
            self.level = 0.0
            return False
        return True

    def depleted(self):
        return self.level <= 0.0

    def fatigued(self):
        return 0.0 < self.level <= self.capacity * self.fatigue_threshold

    def report(self):
        integrity = self.level / self.capacity
        return EnergyState(level=self.level,
                           temperature=1.0 - integrity,
                           integrity=integrity)
