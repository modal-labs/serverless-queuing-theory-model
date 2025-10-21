from datetime import datetime, timedelta
import numpy
from matplotlib import pyplot, dates as mdates
from numpy.lib.stride_tricks import sliding_window_view

pyplot.style.use("modal.mplstyle")


start_date = datetime(2025, 5, 1)
end_date = datetime(2025, 5, 7)

print("Generate time steps")
n_minutes = int((end_date - start_date).total_seconds() / 60)
js = numpy.array(range(n_minutes))
xs = [start_date + timedelta(minutes=i) for i in range(n_minutes)]

print("Generate several months of random noise")
ls = numpy.ones(len(xs)) * 60
ls *= numpy.sin(js / (60 * 24) * 2 * numpy.pi)*0.5 + 1  # multiply by a daily sine wave
ls *= numpy.sin(js / (60 * 24 * 7) * 2 * numpy.pi)*0.2 + 1  # multiply by a weekly sine wave

print("Add a mean-reverting component in log-space (keeps rates positive)")
reversion_strength = 0.001
volatility = 0.03
phi = 1.0 - float(reversion_strength)
series = numpy.zeros(len(xs))
for t in range(1, len(xs)):
    innovation = numpy.random.normal(0.0, volatility)
    series[t] = phi * series[t - 1] + innovation
ls *= numpy.exp(series)

ls *= 1  # Some base rate

print("Create a second level of granularity for the requests")
ls_seconds = numpy.repeat(ls, 60)

print("Turn this into Poisson events")
rs_seconds = numpy.random.poisson(lam=ls_seconds/60)

print("Sum the requests over the minute level of granularity")
rs = rs_seconds.reshape(-1, 60).sum(axis=1)

print("Generate a monthly plot")
pyplot.figure(figsize=(20, 3))
pyplot.plot(xs, rs, linewidth=0.3)
pyplot.xlim(start_date, end_date)
pyplot.xlabel("Timestamp")
pyplot.ylabel("requests per minute")
pyplot.title("Number of requests per minute over a month")
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
pyplot.tight_layout()
pyplot.savefig("month.png", dpi=300)

print("Zoom in on the last day")
pyplot.figure(figsize=(20, 3))
pyplot.plot(xs[-24*60:], rs[-24*60:], linewidth=0.3)
pyplot.xlim(end_date - timedelta(days=1), end_date)
pyplot.xlabel("Timestamp")
pyplot.ylabel("requests per minute")
pyplot.title("Number of requests per minute over a 24h period")
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
pyplot.tight_layout()
pyplot.savefig("day.png", dpi=300)

def pad_zero_left(xs, n):
    return numpy.concatenate([numpy.zeros(n), xs])

def rolling_sum(xs, window_size):
    C = numpy.cumsum(pad_zero_left(xs, window_size - 1))
    return C[window_size-1:] - C[:-window_size+1]

def rolling_min(xs, window_size):
    return numpy.min(sliding_window_view(pad_zero_left(xs, window_size - 1), window_shape=window_size), axis=1)

def rolling_max(xs, window_size):
    return numpy.max(sliding_window_view(pad_zero_left(xs, window_size - 1), window_shape=window_size), axis=1)

print("Compute the number of requests within a sliding window of time")
execution_time = 60
n_busy_containers = rolling_sum(rs_seconds, execution_time)
pyplot.figure(figsize=(20, 3))
pyplot.plot(n_busy_containers)
pyplot.ylabel("number of busy containers")
pyplot.tight_layout()
pyplot.savefig("requests-within-window.png", dpi=300)

print("Compute the number of keepalive containers")
keepalive_time = 60
n_busy_and_draining_containers = rolling_max(n_busy_containers, keepalive_time)
n_draining_containers = n_busy_and_draining_containers - n_busy_containers

print("Compute the number of cold starting containers")
cold_start_time = 60
n_busy_draining_and_coldstarting_containers = rolling_max(n_busy_and_draining_containers, cold_start_time)
n_coldstarting_containers = n_busy_draining_and_coldstarting_containers - n_busy_and_draining_containers

print("Pick a number of buffer containers as the 99th percentile.")
n_buffer_containers = numpy.percentile(n_coldstarting_containers, 99)

print(f"Cap the cold starting containers at the buffer containers ({n_buffer_containers})")
n_coldstarting_containers = numpy.minimum(n_coldstarting_containers, n_buffer_containers)

print("Compute buffer and total containers")
n_buffer_containers = n_buffer_containers - n_coldstarting_containers
n_total_containers = n_busy_containers + n_draining_containers + n_coldstarting_containers + n_buffer_containers

print("Roll up everything by minutes")
n_busy_containers = n_busy_containers.reshape(-1, 60).mean(axis=1)
n_draining_containers = n_draining_containers.reshape(-1, 60).mean(axis=1)
n_coldstarting_containers = n_coldstarting_containers.reshape(-1, 60).mean(axis=1)
n_buffer_containers = n_buffer_containers.reshape(-1, 60).mean(axis=1)
n_total_containers = n_total_containers.reshape(-1, 60).mean(axis=1)

print("Plot the number containers")
pyplot.figure(figsize=(20, 3))
c = numpy.zeros(len(n_busy_containers))
for label, series in [("Busy containers", n_busy_containers), ("Draining containers", n_draining_containers), ("Cold starting containers", n_coldstarting_containers), ("Buffer containers", n_buffer_containers)]:
    pyplot.fill_between(xs, c, c+series, alpha=0.6, label=label)
    c += series
pyplot.legend()
pyplot.xlim(start_date, end_date)
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
pyplot.ylabel("number of containers")
pyplot.tight_layout()
pyplot.savefig("number-containers.png", dpi=300)

print("Plot the utilization rate")
pyplot.figure(figsize=(20, 3))
c = numpy.zeros(len(n_busy_containers))
for label, series in [("Busy containers", n_busy_containers), ("Draining containers", n_draining_containers), ("Cold starting containers", n_coldstarting_containers), ("Buffer containers", n_buffer_containers)]:
    pyplot.fill_between(xs, c / n_total_containers, (c+series)/n_total_containers, alpha=0.6, label=label)
    c += series
avg_utilization_rate = numpy.mean(n_busy_containers) / numpy.mean(n_total_containers)
pyplot.axhline(avg_utilization_rate, color="red", linestyle="--")
pyplot.legend()
pyplot.xlim(start_date, end_date)
pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
pyplot.ylabel("utilization rate")
pyplot.tight_layout()
pyplot.savefig("utilization-rate.png", dpi=300)