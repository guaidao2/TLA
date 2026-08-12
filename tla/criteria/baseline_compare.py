"""TLA vs 标准基线（LSTM / NeuralODE / Mamba 式 SSM）——thrust_cart 同种子同分割对比。

目的：判定"TLA 是否只是换了个皮"——同任务、同数据、同指标，标准 BP 基线 vs TLA（无 BP）。

判据（预注册，跑数前锁死）：
  BC-1 持平或赢（架构有价值）：TLA 未见重力 MSE ≤ 每个基线的 ×1.1
     （对全部基线不劣于 10% 即"持平"；显著更优即"赢"）；
  BC-2 公平性披露：报告各基线训练方式（全部 BP+AdamW）、参数量、训练样本数——
     "TLA 无 BP vs 基线 BP"的范式不对称如实标注（若 TLA 输，归因需无 BP 基线腿，后续）。
诚实预期：无 BP 可能输给 BP——如实记录；输赢都要说清"输/赢在哪"。

基线（全部 BP 训练，torch.nn，可复现）：
  LSTM：nn.LSTM(6→24)→MLP→4（标准循环世界模型）；
  NeuralODE：h'=MLP(h,a)，RK4 积分 dt（连续时间世界模型，手写 RK4，无 torchdiffeq）；
  Mamba 式 SSM：对角状态空间（A diag 稳定参数化，x'=Ax+Bu，y=Cx+Du）——简化 S4 风格。

同种子同分割：与 thrust_cart.run 完全一致（n_traj=30/T=40/n_ep=2，未见重力 g=2.5，
seed_shift 1/99/7 隔离），TLA 与基线吃同一批数据。
"""
import torch
import torch.nn as nn
from tla.model_pr1 import TLAPR1Model
from tla.tasks.thrust_cart import ThrustCartWorld


# ─── 基线模型（全 BP）───
class LSTMWorldModel(nn.Module):
    def __init__(self, in_dim=6, hidden=24, out_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, out_dim))
        self.h = None

    def reset(self):
        self.h = None

    def forward(self, x):
        # x: (1, 6) → (1, 1, 6)
        out, self.h = self.lstm(x.unsqueeze(0), self.h)
        return self.head(out.squeeze(0))


class ODECell(nn.Module):
    """f(h, a) 小网络，RK4 积分 dt（f 输出与状态同维）。"""
    def __init__(self, in_dim=6, hidden=24, out_dim=4, dt=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, out_dim))   # → 状态同维
        self.readout = nn.Sequential(nn.Linear(out_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, out_dim))
        self.dt = dt
        self.h = None

    def reset(self):
        self.h = None

    def _rk4(self, h, a):
        def deriv(hh):
            return self.f(torch.cat([hh, a]))
        k1 = deriv(h)
        k2 = deriv(h + 0.5 * self.dt * k1)
        k3 = deriv(h + 0.5 * self.dt * k2)
        k4 = deriv(h + self.dt * k3)
        return h + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x):
        h0 = x[:4]                                    # 状态部分作为 ODE 初值
        a = x[4:]                                     # 动作作为强制输入
        self.h = self._rk4(h0 if self.h is None else self.h, a)
        return self.readout(self.h)


class SSMWorldModel(nn.Module):
    """Mamba 式对角状态空间（简化 S4）：x' = Ax + Bu, y = Cx + D(s,a)。"""
    def __init__(self, in_dim=6, hidden=24, out_dim=4, dt=0.1):
        super().__init__()
        self.dt = dt
        # 稳定对角 A：Re(λ) < 0（e^{-θ} 参数化）
        self.log_theta = nn.Parameter(torch.zeros(hidden) - 2.0)
        self.B = nn.Linear(in_dim, hidden)
        self.C = nn.Linear(hidden, out_dim)
        self.D = nn.Linear(in_dim, out_dim)
        self.h = None

    def reset(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            self.h = torch.zeros(self.B.out_features)
        A = -torch.exp(self.log_theta)                 # 对角负实部（稳定）
        u = x
        self.h = self.h + self.dt * (A * self.h + self.B(u))
        return self.C(self.h) + self.D(u)


# ─── 训练与评估 ───
def eval_bp(model, trajs):
    """基线（callable nn.Module）评估。"""
    mses = []
    for traj in trajs:
        model.reset()
        for s, a, s_next in traj:
            pred = model(torch.cat([s, a]))
            mses.append(float(torch.mean((pred - s_next) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def eval_tla(model, trajs):
    """TLA（infer 接口）评估。"""
    mses = []
    for traj in trajs:
        model.reset()
        for s, a, s_next in traj:
            pred, _ = model.infer(torch.cat([s, a]))
            mses.append(float(torch.mean((pred - s_next) ** 2).item()))
    return float(torch.tensor(mses).mean().item()) if mses else float("nan")


def train_bp(model, trajs, epochs=2, lr=0.001, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for traj in trajs:
            model.reset()
            loss = torch.tensor(0.0)
            for s, a, s_next in traj:
                pred = model(torch.cat([s, a]))
                loss = loss + torch.mean((pred - s_next) ** 2)
            opt.zero_grad()
            loss.backward()          # 全 BPTT：每轨迹一次 backward
            opt.step()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def run_sensitivity(seed=0, n_traj=16, T=24, epochs_list=(2, 5)):
    """基线训练量敏感性（可复现）：基线给更多 epoch，看是否追上 TLA（排除'没训够'）。"""
    world = ThrustCartWorld(g=1.0, seed=seed)
    train = world.task(g=1.0, n=n_traj, T=T, seed_shift=1)
    test = world.task(g=2.5, n=4, T=20, seed_shift=99)
    out = {}
    for ep in epochs_list:
        row = {}
        for name, cls in (("LSTM", LSTMWorldModel), ("NeuralODE", ODECell),
                          ("SSM", SSMWorldModel)):
            m = cls()
            train_bp(m, train, epochs=ep, seed=seed)
            row[name] = eval_bp(m, test)
        out[ep] = row
    return out


def run(seed=0, verbose=True, n_traj=30, T=40, n_ep=2):
    world = ThrustCartWorld(g=1.0, seed=seed)
    train = world.task(g=1.0, n=n_traj, T=T, seed_shift=1)
    test_other_g = world.task(g=2.5, n=4, T=20, seed_shift=99)   # 未见重力（同 thrust_cart）

    # TLA（无 BP）
    m_tla = TLAPR1Model(obs_dim=6, out_dim=4, seed=seed)
    for _ in range(n_ep):
        for traj in train:
            for s, a, s_next in traj:
                m_tla.train_step(torch.cat([s, a]), s_next)
    mse_tla = eval_tla(m_tla, test_other_g)

    # 基线（BP）
    baselines = {
        "LSTM": LSTMWorldModel(),
        "NeuralODE": ODECell(),
        "SSM(Mamba式)": SSMWorldModel(),
    }
    results = {}
    for name, model in baselines.items():
        train_bp(model, train, epochs=n_ep, seed=seed)
        results[name] = dict(mse=eval_bp(model, test_other_g), params=n_params(model))

    # BC-1：TLA ≤ 每个基线 ×1.1
    p_bc1 = all(mse_tla <= r["mse"] * 1.1 for r in results.values())
    # BC-2 披露（含测试时计算不对称）
    disclosure = dict(tla_bp_free=True, baselines_bp=True,
                      samples=n_traj * T * n_ep,
                      tla_settle_loop="TLA infer 含 settle 迭代（自适应深度，"
                                     "测试时多步精化）；基线单次前向——"
                                     "TLA 用了更多测试时算力，如实披露")

    out = dict(mse_tla=mse_tla, baselines=results, p_bc1=p_bc1,
               disclosure=disclosure)
    if verbose:
        print("=" * 70)
        print(f"TLA vs 基线（thrust_cart，seed={seed}，n_traj={n_traj}/T={T}/n_ep={n_ep}）")
        print("=" * 70)
        print(f"  TLA(无BP):  {mse_tla:.4f}")
        for name, r in results.items():
            rel = mse_tla / max(r["mse"], 1e-12)
            verdict = "TLA赢" if rel < 0.9 else ("持平" if rel <= 1.1 else "TLA输")
            print(f"  {name:>14}(BP): {r['mse']:.4f} (参 {r['params']})  "
                  f"TLA/基线={rel:.2f} → {verdict}")
        print(f"  BC-1 持平或赢（≤1.1×全部基线）: {'PASS' if p_bc1 else 'FAIL'}")
        print(f"  披露: TLA无BP vs 基线BP，样本 {disclosure['samples']}")
        print("=" * 70)
    return out


if __name__ == "__main__":
    run(verbose=True)
