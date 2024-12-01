import numpy as np
import matplotlib.pyplot as plt
from random import uniform

class KArmedBandit:
    def __init__(self, k=1, probs=None):
        self.arms = [uniform(0, 1) for _ in range(k)] if probs is None else [p for p in probs]
    
    def __call__(self, arm):
        return self.pull(arm)
    
    def __getitem__(self, i):
        return self.arms[i]
    
    def __repr__(self):
        name = f"{len(self.arms)}-Armed Bandit with:\n"
        for i, p in enumerate(self.arms):
            name += f"  p{i+1}={p:.3f}\n"
        return name[:-1]
    
    def pull(self, arm):
        if type(arm) != int or len(self.arms) <= arm or arm < 0:
            raise IndexError("Invalid arm selected")
        return int(self.arms[arm] > uniform(0, 1))
    

# Plotting the convergence to true probability
kab = KArmedBandit(probs=[0.2, 0.5, 0.9])
samples = 1200
plt.figure(figsize=(16, 6))
plt.tight_layout()

for arm in range(3):
    rewards = np.asarray([kab(arm) for _ in range(samples)])
    a, b = rewards.cumsum(), (1-rewards).cumsum()
    mean = a/(a+b)
    var = np.sqrt(a*b/((a+b)**2*(a+b+1)))

    plt.plot(mean, label=f"Arm with p={kab[arm]}")
    plt.fill_between(range(samples), mean + var, mean - var, alpha=0.6)
plt.hlines(kab.arms, 0, samples, linestyle="--", colors='dimgray')
plt.legend()
plt.yticks([0, 0.4, 0.6, 0.8, 1] + kab.arms)
plt.title("Convergence to true probability", fontsize=18)
plt.ylabel("Probability p of getting reward r", fontsize=14)
plt.xlabel("Iteration", fontsize=14)
# plt.savefig("convergence.png")
plt.show()


# Comparing Thompson sampling with UCB
k = 3
kab = KArmedBandit(k=k)
params = np.asarray([[1, 1] for _ in range(k)])
print(kab)

regret = np.max(kab.arms) - np.asarray(kab.arms)
arms = list()

# Thompson sampling
iterations = 1000 * k
for i in range(iterations):
    arm = np.argmax([np.random.beta(*ab) for ab in params])
    r = kab(int(arm))
    params[arm] += [r, 1-r]
    arms.append(arm)
    
plt.figure(figsize=(16, 6))
plt.tight_layout()
plt.plot(regret[arms].cumsum(), label="Thompson sampling")

# UCB
for c, style in zip(np.logspace(-3, 0, 4), [(0, (1, 10)), '--', ':', '-.']):
    counts, rewards = [0] * k, [0] * k
    arms = list()
    for i in range(iterations):
        arm = np.argmax([np.infty if count == 0 else r/count + c*np.sqrt(np.log(i)/count)
                         for r, count in zip(rewards, counts)])
        r = kab(int(arm))
        counts[arm] += 1
        rewards[arm] += r
        arms.append(arm)
    plt.plot(regret[arms].cumsum(), label=f"UCB with c={c:.0e}", linestyle=style)
plt.legend()
plt.title("Thompson sampling vs UCB", fontsize=18)
plt.ylabel("Regret", fontsize=14)
plt.xlabel("Iteration", fontsize=14)
# plt.savefig("thompson_ucb.png")
plt.show()