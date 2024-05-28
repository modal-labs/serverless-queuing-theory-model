# Serverless queuing theory model

This implements a simple queuing theory model in a "serverless" setting where we
have the option to scale up and down the number of workers at any point in time.

We make the following assumptions:

* Any worker can be busy, idle, or starting
* Serving time is exponentially distributed with rate parameter R_i
* Startup time is exponentially distributed with rate parameter R_s
* Arrival rate is a Poisson process with rate parameter R_a

This means the state is "memoryless" and it reduces to the following:

1. Total number of starting servers
2. Total number of busy servers
3. Total number of idle servers
4. Current queue size

We can further reduce the state space by noting that it never makes sense to
have idle servers and a queue size at the same time.

These are unbounded, but we can make the state space smaller by capping the numbers.

In each step, we have the option to

1. Add another server
2. Shut down an idle server
3. Do nothing, wait until next quanta

We can encode this policy as a matrix as well.

## Simulating

We simulate this using a Markov chain model, i.e. turn this into a matrix and
perform a power iteration to get the equilibrium state. This represents a probability
distribution over all states of the model.

## Scoring

We can compute the utilization and latency of each model by looking at the
equilibrium distribution and multiplying it with the number of workers and queue size.

## Optimizing this

In order to find the optimal policy, we use a gradient-based optimizer and solve
for the policy that minimizes latency + alpha * waste.

Alpha is an arbitrary multiplier which lets us explore the latency-utilization
tradeoff curve.

## Running this

This is incredibly compute-intensive, so we use Modal and run this on 100s of GPUs
using JAX with GPU support.
