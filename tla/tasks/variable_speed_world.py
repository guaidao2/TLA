"""连续时间变速率世界模型玩具任务（L6 任务形态：踩在 TLA 卖点上）。

- 世界：**阻尼弹簧**（有界振荡）——pos 被 ω² 拉回 0，能量守恒下有界，不是无界漂移；
- 可变速率：弹簧频率 ω 在区间内随机切换（regime switching），dt 不规则采样；
- 观测 = [pos_norm, vel_norm, dt]（带时间戳的世界模型）；
- 目标 = [pos'_norm, vel'_norm]（预测下一观测，标签来自数据流本身——自监督）；
- 测试集用**训练未见过的 ω 区间**（更快振荡 = 未见动态泛化，LTC 的宣称能力）。
- 积分：半隐式欧拉（vel 先更新再 pos），dt·ω < 2 时数值稳定。
"""
import torch


class VariableSpeedWorld:
    def __init__(self, dt_range=(0.05, 0.3), noise=0.02, scale=5.0, seed=0):
        self.dt_range = dt_range
        self.noise = noise
        self.scale = scale
        self.gen = torch.Generator().manual_seed(seed)

    def _rand(self, *s):
        return torch.rand(*s, generator=self.gen)

    def _omega(self, speed_range):
        return self._rand(1).item() * (speed_range[1] - speed_range[0]) + speed_range[0]

    def trajectory(self, T, speed_range):
        pos = (self._rand(1).item() - 0.5) * 2.0 * 3.0      # 初始位置 ±3
        vel = (self._rand(1).item() - 0.5) * 2.0 * 3.0      # 初始速度 ±3
        w = self._omega(speed_range)
        rows = []
        for _ in range(T):
            dt = self._rand(1).item() * (self.dt_range[1] - self.dt_range[0]) + self.dt_range[0]
            if self._rand(1).item() < 0.05:                 # 频率 regime 切换（变速率）
                w = self._omega(speed_range)
            vel = vel - (w ** 2) * pos * dt + (self._rand(1).item() - 0.5) * 2.0 * self.noise
            pos = pos + vel * dt
            rows.append([pos / self.scale, vel / self.scale, dt])
        return torch.tensor(rows, dtype=torch.float32)

    def dataset(self, n_traj, T, speed_range, seed=None):
        """返回 [(obs_t, obs_{t+1})] 列表。"""
        pairs = []
        for traj in self.trajectories(n_traj, T, speed_range, seed):
            for t in range(T - 1):
                pairs.append((traj[t], traj[t + 1]))
        return pairs

    def trajectories(self, n_traj, T, speed_range, seed=None):
        """返回 [traj] 列表，traj 为 T×3 张量（按轨迹边界可分，供 LTC 身体重置）。"""
        gen_state = self.gen.get_state()
        if seed is not None:
            self.gen = torch.Generator().manual_seed(seed)
        out = [self.trajectory(T, speed_range) for _ in range(n_traj)]
        self.gen.set_state(gen_state)
        return out
