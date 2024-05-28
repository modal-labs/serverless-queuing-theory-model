import io
import modal

app = modal.App()
image = modal.Image.debian_slim().pip_install("jax[cuda12]", "scipy", "matplotlib")

with image.imports():
    from jax import grad, numpy, jit
    from scipy.optimize import minimize


class Params:
    """Various matrices & arrays needed by the code."""

    def __init__(self, Rb, Rs, Ra, eps, max_n_workers=15, max_queue_size=45, queue_full_penalty=30.0):
        self.Rb = Rb
        self.Rs = Rs
        self.Ra = Ra
        self.eps = eps
        self.max_n_workers = max_n_workers
        self.max_queue_size = max_queue_size
        self.queue_full_penalty = queue_full_penalty
        self.state_space_indices = {}
        for b, i, s, q in self.iterate_indices():
            self.state_space_indices[(b, i, s, q)] = len(self.state_space_indices)
        self.S = len(self.state_space_indices)
        print("Total size of state space:", self.S)

    def iterate_indices(self):
        for b in range(self.max_n_workers + 1):  # busy
            for i in range(self.max_n_workers + 1):  # idle
                for s in range(self.max_n_workers + 1):  # starting
                    for q in range(self.max_queue_size + 1):  # queue size
                        if b + i + s > self.max_n_workers:
                            continue
                        if i > 0 and q > 0:  # Never idle workers and queue size
                            continue
                        yield b, i, s, q


class Data:
    def __init__(self, params):
        # Precompute state space

        def get_index(b, i, s, q):
            return params.state_space_indices[(b, i, s, q)]

        # Precompute transition probabilities
        M = [[0.0] * params.S for i in range(params.S)]
        for b, i, s, q in params.iterate_indices():
            z = get_index(b, i, s, q)
            if b > 0 and q > 0:
                # Busy worker finishes, there are more items in the queue
                z2 = get_index(b, i, s, q - 1)
                M[z][z2] = params.Rb * params.eps * b
            if b > 0 and q == 0:
                # Busy worker finishes, add an idle worker
                z2 = get_index(b - 1, i + 1, s, q)
                M[z][z2] = params.Rb * params.eps * b
            if s > 0 and q > 0:
                # Starting worker ready, pick up a task
                z2 = get_index(b + 1, i, s - 1, q - 1)
                M[z][z2] = params.Rs * params.eps * s
            if s > 0 and q == 0:
                # Starting worker ready, add an idle worker
                z2 = get_index(b, i + 1, s - 1, q)
                M[z][z2] = params.Rs * params.eps * s
            if i > 0:
                # New task arrives, idle worker picks up new task
                z2 = get_index(b + 1, i - 1, s, q)
                M[z][z2] = params.Ra * params.eps
            if i == 0 and q < params.max_queue_size:
                # New task arrives, goes to queue
                z2 = get_index(b, i, s, q + 1)
                M[z][z2] = params.Ra * params.eps

        M = numpy.array(M)

        # Turn rates into probabilities (already approx correct for small values)
        M = 1 - numpy.exp(-M)

        # Add diagonal elements for staying
        self.M = M + numpy.diag(1 - M.sum(axis=1))
        assert self.M.min() >= 0.0 and self.M.max() <= 1.0

        # Precompute matrices for (u) starting a new worker (d) shutting down an idle worker
        Pu = [[0.0] * params.S for i in range(params.S)]
        Pd = [[0.0] * params.S for i in range(params.S)]
        for b, i, s, q in params.iterate_indices():
            z = get_index(b, i, s, q)
            if b + i + s + q < params.max_n_workers:
                z2 = get_index(b, i, s + 1, q)
                Pu[z][z2] = 1.0
            if i > 0:
                z2 = get_index(b, i - 1, s, q)
                Pd[z][z2] = 1.0

        Pu = numpy.array(Pu)
        Pd = numpy.array(Pd)
        self.Pu = Pu + numpy.diag(1 - Pu.sum(axis=1))
        self.Pd = Pd + numpy.diag(1 - Pd.sum(axis=1))

        # Precompute number of each thing
        Nb = [0] * params.S
        Ni = [0] * params.S
        Ns = [0] * params.S
        Nq = [0] * params.S
        Nf = [0] * params.S  # "system full" penalization
        for b, i, s, q in params.iterate_indices():
            z = get_index(b, i, s, q)
            Nb[z] = b
            Ni[z] = i
            Ns[z] = s
            Nq[z] = q

        # Penalize the state when we run out of busy workers and have to reject queue items
        for b, i, s, q in params.iterate_indices():
            if q == 0 and i > 0 and b + i + s == params.max_n_workers:
                z = get_index(b, i, s, 0)
                Nf[z] = 1

        # Penalize the state when the queue is full so we have to reject new queue items
        for b, i, s, q in params.iterate_indices():
            if q == params.max_queue_size:
                Nf[z] = 1

        self.Nb = numpy.array(Nb)
        self.Ni = numpy.array(Ni)
        self.Ns = numpy.array(Ns)
        self.Nq = numpy.array(Nq)
        self.Nf = numpy.array(Nf)


def simulate(params, data, P, u, d, steps=1000):
    Au = numpy.dot(numpy.diag(u), data.Pu) + numpy.diag(1 - u)
    Ad = numpy.dot(numpy.diag(d), data.Pd) + numpy.diag(1 - d)

    A = numpy.dot(Au, Ad)
    N = numpy.dot(data.M, A)
    # assert N.min() >= 0.0 and N.max() <= 1.0

    for i in range(steps):
        P = numpy.dot(P, N)

        if i % 100 == 99:
            P /= P.sum()  # Norm should be 1.0 but this is needed for numerical stability

    return P


def optimize(params, data, P, alpha):
    def objective(ud):
        u = ud[: params.S]
        d = ud[params.S :]

        P2 = simulate(params, data, P, u, d)
        queue_size = numpy.dot(P2, data.Nq)
        waste = numpy.dot(P2, data.Nb + data.Ni + data.Ns + data.Nf * params.queue_full_penalty)
        return queue_size + alpha * waste

    ud0 = numpy.ones(2 * params.S) * 1e-3
    bounds = [(1e-3, 1.0)] * 2 * params.S

    ret = minimize(jit(objective), ud0, jac=jit(grad(objective)), bounds=bounds)
    return ret.x[: params.S], ret.x[params.S :]


@app.function(gpu="A100", image=image, timeout=900)
def simulate_alpha(params, P, alpha):
    data = Data(params)

    u, d = optimize(params, data, P, alpha)
    P2 = simulate(params, data, P, u, d)
    return P2


@app.function(gpu="A100", image=image, timeout=3600)
def compute_tradeoffs(params):
    data = Data(params)

    # Generate initial probability distribution
    P = numpy.ones(params.S) / params.S

    # Simulate extra many steps first
    u = numpy.ones(params.S) * 1e-2
    d = numpy.ones(params.S) * 1e-2

    print("Simulating lots of steps")
    P = simulate(params, data, P, u, d, steps=10000)

    print("Solving with alpha=1")
    u, d = optimize(params, data, P, alpha=1.0)

    print("Simulating again")
    P = simulate(params, data, P, u, d, steps=10000)

    # Compute a bunch of optimal policies trading off latency vs utilization
    tradeoff_curve = []
    alphas = numpy.exp(numpy.linspace(-7, 7, 50))
    args = [(params, P, alpha) for alpha in alphas]
    for P2 in simulate_alpha.starmap(args):
        queue_size = numpy.dot(P2, data.Nq)
        waste = numpy.dot(P2, data.Nb + data.Ni + data.Ns + data.Nf)
        latency = numpy.dot(P2, data.Nq) / params.Ra
        utilization = numpy.dot(P2, data.Nb) / numpy.dot(P2, data.Nb + data.Ni + data.Ns)
        print(f"{queue_size=:.4f} {waste=:.4f} {latency=:.4f} {utilization=:.4f}")

        l = float(latency)
        u = float(utilization)
        tradeoff_curve.append((l, u))

    # Remove points not on the Pareto frontier
    tradeoff_curve_2 = []
    for l, u in tradeoff_curve:
        if not any(l2 < l and u2 > u for l2, u2 in tradeoff_curve):
            tradeoff_curve_2.append((l, u))

    tradeoff_curve_2.sort()
    return tradeoff_curve_2


@app.function(image=image, timeout=3600)
def plot(plot_params):
    from matplotlib import pyplot, ticker

    pyplot.style.use("ggplot")
    for Rs, tradeoff_curve in plot_params:
        ls = [l for l, u in tradeoff_curve]
        us = [100 * u for l, u in tradeoff_curve]
        pyplot.plot(ls, us, label=f"{Rs=}")

    pyplot.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    pyplot.legend()
    pyplot.xlabel("Latency (s)")
    pyplot.ylabel("Utilization (%)")

    pyplot.tight_layout()
    buf = io.BytesIO()
    pyplot.savefig(buf, dpi=300)
    return buf.getvalue()


@app.local_entrypoint()
def run():
    Rb = 1.0  # busy -> finish rate (s^-1)
    Rs = 1.0  # starting -> ready rate (s^-1)
    Ra = 1.0  # arrival rate (s^-1)
    eps = 0.01  # time quanta

    params = []
    for Rs in [0.1, 0.3, 1.0, 3.0, 10.0]:
        params.append(Params(Rb, Rs, Ra, eps))

    plot_data = []
    for params, (tradeoff_curve) in zip(params, compute_tradeoffs.map(params)):
        plot_data.append((params.Rs, tradeoff_curve))

    png_data = plot.remote(plot_data)
    with open("tradeoff.png", "wb") as f:
        f.write(png_data)


# for b, i, s, q in iterate_indices():
#    z = get_index(b, i, s, q)
#    print(f"{b} {i} {s} {q} p: {P2[z]:6.2%} up: {u[z]:9.6f} down: {d[z]:9.6f}")
#    for b2, i2, s2, q2 in iterate_indices():
#        z2 = get_index(b2, i2, s2, q2)
#        if M[z][z2] > 0:
#            print(f"   -> {b2} {i2} {s2} {q2} p: {M[z][z2]}")
