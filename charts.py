from datetime import datetime, timedelta
import numpy
from matplotlib import pyplot, dates as mdates

pyplot.style.use("ggplot")

# Start with just Poisson noise
xs = [datetime(2024, 5, 22) + timedelta(minutes=i) for i in range(60 * 24)]
rs = numpy.random.poisson(lam=60.0, size=len(xs))
pyplot.figure(figsize=(16, 4))
pyplot.plot(xs, rs, linewidth=0.3)
pyplot.xlim(datetime(2024, 5, 22), datetime(2024, 5, 23))
pyplot.ylim(0, 90)
pyplot.xlabel("Time")
pyplot.ylabel("reqs/min")
pyplot.title("Number of requests per minute over a 24h period")
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
pyplot.tight_layout()
pyplot.savefig("static-day.png")

# Do a sine wave now
xs = [datetime(2024, 5, 22) + timedelta(minutes=i) for i in range(3 * 60 * 24)]
ls = numpy.ones(len(xs)) * 60
ls *= numpy.sin(numpy.linspace(0, 3 * 2 * numpy.pi, len(xs), endpoint=False))*0.5 + 1  # multiply by a daily sine wave
rs = numpy.random.poisson(lam=ls, size=len(xs))
pyplot.figure(figsize=(16, 4))
pyplot.plot(xs, rs, linewidth=0.3)
pyplot.xlim(datetime(2024, 5, 22), datetime(2024, 5, 25))
pyplot.ylim(0, 120)
pyplot.xlabel("Time")
pyplot.ylabel("reqs/min")
pyplot.title("Number of requests per minute over a 3 day period")
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
pyplot.tight_layout()
pyplot.savefig("sine-wave-day.png")

# Now do a separate one with over a month
xs = [datetime(2024, 5, 1) + timedelta(minutes=i) for i in range(60 * 24 * 31)]
ls = numpy.ones(len(xs)) * 60
ls *= numpy.sin(numpy.linspace(0, 31 * 2 * numpy.pi, len(xs), endpoint=False))*0.5 + 1  # multiply by a daily sine wave
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

# Generate a monthly plot
pyplot.figure(figsize=(16, 4))
pyplot.plot(xs, rs, linewidth=0.3)
pyplot.xlim(datetime(2024, 5, 1), datetime(2024, 6, 1))
pyplot.xlabel("Timestamp")
pyplot.ylabel("reqs/min")
pyplot.title("Number of requests per minute over a month")
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
pyplot.tight_layout()
pyplot.savefig("dynamic-month.png")

# Zoom in on the last day
pyplot.figure(figsize=(16, 4))
pyplot.plot(xs[-24*60:], rs[-24*60:], linewidth=0.3)
pyplot.xlim(datetime(2024, 5, 31), datetime(2024, 6, 1))
pyplot.xlabel("Timestamp")
pyplot.ylabel("reqs/min")
pyplot.title("Number of requests per minute over a 24h period")
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:M:%S"))
pyplot.tight_layout()
pyplot.savefig("dynamic-day.png")
