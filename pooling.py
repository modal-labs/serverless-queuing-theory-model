from matplotlib import pyplot
import numpy
from datetime import datetime, timedelta

pyplot.style.use("modal.mplstyle")

xs = [datetime(2024, 5, 1) + timedelta(minutes=i) for i in range(60 * 24 * 31)]

def generate_one():
    ls = numpy.ones(len(xs)) * 60
    ls *= numpy.sin(numpy.linspace(0, 31 * 2 * numpy.pi, len(xs), endpoint=False))*0.3 + 1  # multiply by a daily sine wave
    ls *= numpy.sin(numpy.linspace(0, 31/7 * 2 * numpy.pi, len(xs), endpoint=False))*0.2 + 1  # multiply by a weekly sine wave
    ls *= numpy.exp(numpy.cumsum(numpy.random.normal(loc=3e-5, scale=1e-4, size=len(xs))))  # multiply by an exponential random drift
    for i in range(3):
        # Add some random traffic spikes
        j = numpy.random.randint(0, len(xs)-1)
        mag = 2 * numpy.exp(-i * 0.4)
        ls *= numpy.array([1 + mag * numpy.exp(-5e-4 * (k - j if k > j else (k - j)**2)) for k in range(len(xs))])
    ls *= numpy.array([1.5 if 0.2 <= k / len(xs) <= 0.5 else 1 for k in range(len(xs))])  # Add a fake "backfill"
    for ws in [10, 60, 180]:
        # Add an autoregressive thing in a dumb way
        zs = numpy.random.normal(loc=0, scale=1e-2, size=len(xs)+ws)
        ls *= numpy.exp(numpy.cumsum(zs[ws:]) - numpy.cumsum(zs[:-ws]))

    rs = numpy.random.poisson(lam=ls)
    n_gpus = []
    for i in range(rs.shape[0]):
        n_gpus.append(max(rs[max(0, i-10):i+1]))
    return numpy.array(n_gpus)

n_users = 5
n_gpus = [generate_one() for i in range(n_users)]

colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']

fig, axes = pyplot.subplots(1, 2, sharey=True, figsize=(16, 8))
peak = 0
s = n_gpus[0] * 0
for i, n in enumerate(n_gpus):
    axes[0].fill_between(xs, peak + n * 0, peak + n, label=f"User {i+1} GPUs", color=colors[i])
    axes[0].fill_between(xs, peak + n, peak + max(n), color=colors[i], alpha=0.2)
    peak += max(n)
    axes[1].fill_between(xs, s, s + n, color=colors[i], label=f"User {i+1} GPUs")
    s += n

axes[1].fill_between(xs, s, s*0 + max(s), alpha=0.2, label="Pool idle GPUs")
axes[0].legend()
axes[1].legend()
axes[0].set_title(f"Single-tenant pools with {peak} GPUS")
axes[1].set_title(f"Multi-tenant pools with {max(s)} GPUS")
axes[0].set_ylabel("Number of GPUs")

axes[0].set_ylim([0, None])
pyplot.tight_layout()
pyplot.savefig("pooling.png", dpi=300)
