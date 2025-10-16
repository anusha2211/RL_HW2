# main.py
from policy import *
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=100, threshold=np.inf)

if __name__ == "__main__":

    neurons_per_layer = (3, 2)

    policy = PolicyNet(neurons_per_layer=neurons_per_layer)
    print(f"\nNetwork architecture: {neurons_per_layer}")
    
    theta        = policy.get_policy_parameters()
    n_parameters = theta.size
    print(f"Total number of parameters: {n_parameters}")
    print(f"\nPolicy parameters:\n{theta}")

    state = np.array([0.9, -0.06])
    print(f"\nState: {state}")
    
    action = policy.act(state)
    print(f"\nAction selected by the policy: {action}")

    noise = np.random.standard_normal(n_parameters).astype(np.float32)
    print(f"\nNoise that will be added to the policy parameters:\n{noise}")

    new_theta = theta + noise
    policy.set_policy_parameters(new_theta)    
    print(f"\nNew policy parameters:\n{policy.get_policy_parameters()}")

    action = policy.act(state)
    print(f"\nAction selected by the updated policy: {action}")


# ES DEMO (uses shared es_trainer.py)
import matplotlib.pyplot as plt
from es_trainer import ESConfig, train_es

def run_es_demo():
    cfg = ESConfig(
        neurons_per_layer=(5,),
        temperature=0.1,
        episodes_per_eval=15,
        iterations=100,
        master_seed=42,
        population=60,
        sigma=0.25,
        alpha=0.10,
        use_topk=True,    #True = top-K (PDF), False = rank-based
        elites=5,
        parallel=True,
        num_workers=None,
        use_shaping=False,
    )
    out = train_es(cfg)
    hist = out["history"]
    xs = [h["iter"] for h in hist]
    ys = [h["J_curr"] for h in hist]
    yb = [h["J_best_true"] for h in hist]

    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, label="Current policy (avg return)")
    plt.plot(xs, yb, label="Best candidate (per iter)", alpha=0.75)
    plt.axhline(-150, linestyle="--", label="Baseline-acceptable (~-150)")
    plt.axhline(-120, linestyle="--", label="Near-optimal (~-120)")
    plt.xlabel("ES iteration"); plt.ylabel("Average return (higher is better)")
    plt.title("ES on Mountain Car (PolicyNet, PyTorch)")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    print("\n[ES Demo] Starting ES training with PolicyNet (PyTorch) ...")
    run_es_demo()
