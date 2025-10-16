# es_trainer.py
# Shared ES trainer using PolicyNet (PyTorch) and MountainCarEnv (NumPy).

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count

from env import MountainCarEnv, MAX_POSITION, TIMEOUT
from policy import PolicyNet

# ----------------- Rollout & evaluation helpers -----------------

def rollout_once(env_seed: int,
                 theta: np.ndarray,
                 policy_arch: Tuple[int, ...],
                 temperature: float = 0.1,
                 max_steps: int = TIMEOUT) -> tuple[float, bool, float]:
    """
    Run one episode; return (sum of rewards, reached_goal?, max_position).
    """
    env = MountainCarEnv(seed=env_seed)
    pi  = PolicyNet(neurons_per_layer=policy_arch)
    pi.temperature = float(temperature)
    pi.set_policy_parameters(theta)

    s = env.reset()
    G = 0.0
    max_x = float(s[0])
    reached = False
    for _ in range(max_steps):
        a = pi.act(s)                    # {-1.0, 0.0, +1.0}
        s, r, done, _ = env.step(int(a)) # env expects int {-1,0,+1}
        G += r
        if s[0] > max_x:
            max_x = float(s[0])
        if s[0] >= MAX_POSITION:
            reached = True
        if done:
            break
    return float(G), bool(reached), float(max_x)

def evaluate_theta(theta: np.ndarray,
                   policy_arch: Tuple[int, ...],
                   temperature: float,
                   base_seed: int,
                   n_episodes: int) -> tuple[float, float, float]:
    """
    Mean across n episodes: (mean_return, success_rate, mean_max_x).
    """
    returns, succ, maxx = [], 0, []
    for e in range(n_episodes):
        G, ok, mx = rollout_once(base_seed + e, theta, policy_arch, temperature)
        returns.append(G); maxx.append(mx); succ += int(ok)
    return float(np.mean(returns)), succ / n_episodes, float(np.mean(maxx))

def _centered_ranks(x: np.ndarray) -> np.ndarray:
    """
    Rank-based weighting in [-0.5, 0.5] for variance reduction.
    """
    if len(x) <= 1:
        return np.array([0.0], dtype=np.float64)
    r = x.argsort().argsort().astype(np.float64)
    return r / (len(x) - 1) - 0.5

def _worker_eval(args):
    theta, arch, temp, seed, neps = args
    return evaluate_theta(theta, arch, temp, seed, neps)

#ES config & trainer

@dataclass
class ESConfig:
    # Policy / eval
    neurons_per_layer: Tuple[int, ...] = (5,)
    temperature: float = 0.1
    episodes_per_eval: int = 15
    iterations: int = 100
    master_seed: int = 42

    # ES hyperparams
    population: int = 60        # P
    sigma: float = 0.25         # exploration scale
    alpha: float = 0.10         # step size

    # Update rule
    use_topk: bool = True       # True => top-K; False => rank-based ES
    elites: int = 5             # K (only used if use_topk=True)

    # Optional shaping (training-time only)
    use_shaping: bool = False
    beta: float = 100.0
    warmup_shaping_iters: int = 0  # 0 ⇒ shaping active for all iters when use_shaping=True

    # Parallel evaluation
    parallel: bool = True
    num_workers: Optional[int] = None  # None ⇒ cpu_count()-1

def train_es(cfg: ESConfig) -> Dict[str, Any]:
    """
    Train ES given ESConfig. Returns:
      dict(theta=..., policy=PolicyNet, history=[{iter, J_curr, SR_curr, MX_curr, fitness_* , J_best_true}], config=cfg.__dict__)
    """
    rng = np.random.default_rng(cfg.master_seed)

    # Probe policy to get dimensionality
    probe = PolicyNet(neurons_per_layer=cfg.neurons_per_layer)
    probe.temperature = float(cfg.temperature)
    theta = probe.get_policy_parameters().astype(np.float64)
    dim = theta.size

    # Baseline for logging
    J0, SR0, MX0 = evaluate_theta(theta, cfg.neurons_per_layer, cfg.temperature,
                                  base_seed=10_000, n_episodes=cfg.episodes_per_eval)
    print(f"[ES] init: J={J0:.1f}  SR={SR0:.0%}  MX={MX0:.3f}")

    history: List[Dict[str, float]] = []

    # Setup pool
    pool = None
    if cfg.parallel:
        nprocs = cfg.num_workers or max(1, cpu_count() - 1)
        pool = Pool(processes=nprocs)
        print(f"[ES] parallel ON with {nprocs} workers")

    for it in range(cfg.iterations):
        sigma_t = cfg.sigma
        alpha_t = cfg.alpha

        # Mirrored sampling
        half = cfg.population // 2
        eps_half = rng.standard_normal(size=(half, dim))
        eps_all = np.vstack([eps_half, -eps_half])
        if eps_all.shape[0] < cfg.population:  # odd P
            eps_all = np.vstack([eps_all, rng.standard_normal(size=(1, dim))])

        thetas = [theta + sigma_t * eps_all[i] for i in range(cfg.population)]

        # Evaluate population
        if pool is None:
            results = [evaluate_theta(th, cfg.neurons_per_layer, cfg.temperature,
                                      base_seed=20_000 + it*1_000 + i*100,
                                      n_episodes=cfg.episodes_per_eval)
                       for i, th in enumerate(thetas)]
        else:
            args = [(thetas[i], cfg.neurons_per_layer, cfg.temperature,
                     20_000 + it*1_000 + i*100, cfg.episodes_per_eval)
                    for i in range(cfg.population)]
            results = pool.map(_worker_eval, args, chunksize=max(1, cfg.population // ((pool._processes or 1)*4)))

        cand_J  = np.array([r[0] for r in results], dtype=np.float64)
        cand_MX = np.array([r[2] for r in results], dtype=np.float64)

        # (Optional) shaping
        use_shape_now = cfg.use_shaping and (cfg.warmup_shaping_iters == 0 or it < cfg.warmup_shaping_iters)
        fitness = cand_J + cfg.beta * cand_MX if use_shape_now else cand_J.copy()

        # Evaluate current policy (for logging)
        J_curr, SR_curr, MX_curr = evaluate_theta(theta, cfg.neurons_per_layer, cfg.temperature,
                                                  base_seed=30_000 + it*1_000,
                                                  n_episodes=cfg.episodes_per_eval)

        # Search direction
        if cfg.use_topk:
            # Top-K ES (as in the PDF): baseline-subtracted weights on the top-K epsilons
            idx = np.argsort(cand_J)         # ascending
            top = idx[-cfg.elites:]
            diffs = (cand_J[top] - J_curr)   # shape (K,)
            grad = (diffs[:, None] * eps_all[top]).sum(axis=0) / (sigma_t * cfg.elites)
        else:
            # Rank-based ES: weights for all epsilons via centered ranks
            weights = _centered_ranks(fitness)  # shape (P,)
            grad = (weights[:, None] * eps_all).sum(axis=0) / (sigma_t * cfg.population)

        # Normalize and update
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad = grad / norm
        theta = theta + alpha_t * grad

        # Log
        elite_mean_fit = float(np.mean(np.sort(fitness)[-max(1, min(cfg.population, cfg.elites)):]))
        best_fit  = float(np.max(fitness))
        best_true = float(np.max(cand_J))
        history.append(dict(
            iter=it,
            J_curr=float(J_curr),
            SR_curr=float(SR_curr),
            MX_curr=float(MX_curr),
            fitness_elite_mean=elite_mean_fit,
            fitness_best=best_fit,
            J_best_true=best_true,
        ))

        if it % 5 == 0 or it == cfg.iterations - 1:
            print(f"[ES] it={it:3d}  J={J_curr:.1f}  SR={SR_curr:.0%}  MX={MX_curr:.3f}  best_true={best_true:.1f}")

    if pool is not None:
        pool.close(); pool.join()

    # Final policy (loaded with theta)
    final_policy = PolicyNet(neurons_per_layer=cfg.neurons_per_layer)
    final_policy.temperature = float(cfg.temperature)
    final_policy.set_policy_parameters(theta)

    return dict(theta=theta, policy=final_policy, history=history, config=cfg.__dict__)

# Convenience: run one (returns history list)
def run_one(cfg: ESConfig) -> List[Dict[str, float]]:
    out = train_es(cfg)
    return out["history"]
