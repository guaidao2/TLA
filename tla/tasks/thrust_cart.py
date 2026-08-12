"""2D 推力体世界模型（连续控制，本地生成，无外部依赖）。

比弹簧世界更真实的持续学习场地：
- 状态 s = (x, y, vx, vy)（4 维，位置在 [0,1]² 内反弹有界）；
- 动作 a = (ax, ay)（2 维推力，U(−1,1) 随机采样或恒推）；
- 动力学：半隐式速度更新（含重力 g，可调）+ 位置边界反弹（restitution 0.9）
  + 速度 clamp [−vmax, vmax]——非线性、含动作通道、物理参数（重力）可做任务序列；
- 任务序列：task(g, n, T)——不同重力 = 不同物理规则 = 持续学习（A: g=1.0 → B: g=2.0）。

与弹簧世界（3 维纯观测预测）的差异：6 维输入（s+a）、预测 4 维 s'、
非线性（重力+反弹）、动作条件化（模型必须真正使用动作才能学好）。
"""
import torch


class ThrustCartWorld:
    def __init__(self, g=1.0, dt=0.1, vmax=1.0, bbox=(0.0, 1.0),
                 restitution=0.9, seed=0):
        self.g = g
        self.dt = dt
        self.vmax = vmax
        self.bbox = bbox
        self.restitution = restitution
        self.rng = torch.Generator().manual_seed(seed)

    def _u(self, lo, hi):
        """[lo, hi) 均匀采样（用 rng，确定性）。"""
        return torch.empty(1).uniform_(lo, hi, generator=self.rng).item()

    def _step(self, s, a):
        """s=(x,y,vx,vy), a=(ax,ay) → s'（半隐式速度 + 边界反弹）。"""
        x, y, vx, vy = s.tolist()
        ax, ay = a.tolist()
        # 速度更新（含重力，clamp 有界）
        vx1 = min(max(vx + ax * self.dt, -self.vmax), self.vmax)
        vy1 = min(max(vy + (ay - self.g) * self.dt, -self.vmax), self.vmax)
        # 位置更新 + 边界反弹（restitution；反弹后速度再 clamp，防 restitution>1 发散）
        x1 = x + vx1 * self.dt
        y1 = y + vy1 * self.dt
        lo, hi = self.bbox
        if x1 < lo:
            x1, vx1 = 2 * lo - x1, -vx1 * self.restitution
        elif x1 > hi:
            x1, vx1 = 2 * hi - x1, -vx1 * self.restitution
        if y1 < lo:
            y1, vy1 = 2 * lo - y1, -vy1 * self.restitution
        elif y1 > hi:
            y1, vy1 = 2 * hi - y1, -vy1 * self.restitution
        vx1 = min(max(vx1, -self.vmax), self.vmax)
        vy1 = min(max(vy1, -self.vmax), self.vmax)
        return torch.tensor([x1, y1, vx1, vy1], dtype=torch.float32)

    def trajectory(self, T=40, act_mode="random", fixed_act=None):
        """返回 [(s_t, a_t, s_{t+1})]（T 个样本）。act_mode: random | fixed。"""
        assert not (act_mode == "fixed" and fixed_act is None), \
            "act_mode='fixed' 必须提供 fixed_act"
        s = torch.tensor(
            [self._u(self.bbox[0], self.bbox[1]),
             self._u(self.bbox[0], self.bbox[1]),
             self._u(-self.vmax, self.vmax),
             self._u(-self.vmax, self.vmax)],
            dtype=torch.float32)
        out = []
        for _ in range(T):
            if act_mode == "fixed":
                a = torch.tensor(fixed_act, dtype=torch.float32)
            else:
                a = torch.tensor([self._u(-1.0, 1.0), self._u(-1.0, 1.0)],
                                 dtype=torch.float32)
            s_next = self._step(s, a)
            out.append((s.clone(), a.clone(), s_next.clone()))
            s = s_next
        return out

    def trajectories(self, n, T=40, act_mode="random", seed_shift=0):
        """n 条轨迹（seed 由 self.rng 推进，seed_shift 用于测试集隔离）。"""
        if seed_shift:
            rng = torch.Generator().manual_seed(
                int(torch.randint(0, 2 ** 31, (1,), generator=self.rng).item())
                + seed_shift)
            self.rng = rng
        return [self.trajectory(T=T, act_mode=act_mode) for _ in range(n)]

    def task(self, g, n, T=40, act_mode="random", seed_shift=0):
        """任务序列接口：以指定重力 g 生成 n 条轨迹（不同物理规则）。"""
        return ThrustCartWorld(g=g, dt=self.dt, vmax=self.vmax,
                               bbox=self.bbox, restitution=self.restitution,
                               seed=int(torch.randint(0, 2 ** 31, (1,),
                                                      generator=self.rng).item())
                               + seed_shift).trajectories(n, T, act_mode)


def flatten(trajs):
    """轨迹列表 → (s_t, a_t, s_next) 样本列表。"""
    xs, acts, ts = [], [], []
    for traj in trajs:
        for s, a, s_next in traj:
            xs.append(s)
            acts.append(a)
            ts.append(s_next)
    return xs, acts, ts
