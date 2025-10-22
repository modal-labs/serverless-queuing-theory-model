// Core utilities mirroring cost_estimator.py

function range(n) {
  return Array.from({ length: n }, (_, i) => i);
}

function dateRangeMinutes(start, end) {
  const minutes = Math.floor((end - start) / 60000);
  return range(minutes).map(i => new Date(start.getTime() + i * 60000));
}

function repeatEach(xs, times) {
  const out = new Float64Array(xs.length * times);
  for (let i = 0; i < xs.length; i++) {
    const v = xs[i];
    for (let j = 0; j < times; j++) out[i * times + j] = v;
  }
  return out;
}

// Random helpers
function makeRng(seed) {
  // Mulberry32 PRNG; if seed is undefined/null, fall back to Math.random
  if (seed === undefined || seed === null) return () => Math.random();
  let a = (seed >>> 0) || 0;
  return function() {
    a += 0x6D2B79F5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn() {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function poisson(lam) {
  // Knuth's algorithm — fine for small lam (here lam<=1 most of the time)
  const L = Math.exp(-lam);
  let k = 0;
  let p = 1;
  do { k++; p *= Math.random(); } while (p > L);
  return k - 1;
}

function randnWith(random) {
  // Box-Muller using provided RNG
  let u = 0, v = 0;
  while (u === 0) u = random();
  while (v === 0) v = random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function poissonWith(lam, random) {
  // Knuth using provided RNG
  const L = Math.exp(-lam);
  let k = 0;
  let p = 1;
  do { k++; p *= random(); } while (p > L);
  return k - 1;
}

// Rolling ops
function padZeroLeft(xs, n) {
  const out = new Float64Array(xs.length + n);
  // first n are already 0
  out.set(xs, n);
  return out;
}

function rollingSum(xs, window) {
  const padded = padZeroLeft(xs, window - 1);
  const cumsum = new Float64Array(padded.length);
  let running = 0;
  for (let i = 0; i < padded.length; i++) {
    running += padded[i];
    cumsum[i] = running;
  }
  const out = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    out[i] = cumsum[i + window - 1] - (i > 0 ? cumsum[i - 1] : 0);
  }
  return out;
}

function rollingMax(xs, window) {
  // sliding window min/max via deque (O(n))
  const n = xs.length;
  const out = new Float64Array(n);
  const deq = []; // store indices
  // prefill with window-1 zeros on left
  const padded = padZeroLeft(xs, window - 1);
  for (let i = 0; i < padded.length; i++) {
    const val = padded[i];
    while (deq.length) {
      const backVal = padded[deq[deq.length - 1]];
      if (backVal <= val) deq.pop(); else break;
    }
    deq.push(i);
    const start = i - window + 1;
    if (start >= 0) {
      while (deq[0] < start) deq.shift();
      const outIdx = start;
      out[outIdx] = padded[deq[0]];
    }
  }
  return out;
}

function reshapeMeanPerMinute(xsSeconds) {
  // assumes length divisible by 60
  const minutes = xsSeconds.length / 60;
  const out = new Float64Array(minutes);
  for (let m = 0; m < minutes; m++) {
    let s = 0;
    for (let j = 0; j < 60; j++) s += xsSeconds[m * 60 + j];
    out[m] = s / 60.0;
  }
  return out;
}

function reshapeSumPerMinute(xsSeconds) {
  const minutes = xsSeconds.length / 60;
  const out = new Float64Array(minutes);
  for (let m = 0; m < minutes; m++) {
    let s = 0;
    for (let j = 0; j < 60; j++) s += xsSeconds[m * 60 + j];
    out[m] = s;
  }
  return out;
}

function generateData({ startDate, endDate, seed }) {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const nMinutes = Math.floor((end - start) / 60000);
  const xsMinutes = dateRangeMinutes(start, end);
  const js = range(nMinutes);
  const random = makeRng(seed);

  // base rate with daily and weekly sin components
  let ls = new Float64Array(nMinutes).fill(20);
  for (let i = 0; i < nMinutes; i++) {
    const daily = Math.sin((js[i] / (60 * 24)) * 2 * Math.PI) * 0.67 + 1;
    const weekly = Math.sin((js[i] / (60 * 24 * 7)) * 2 * Math.PI) * 0.2 + 1;
    ls[i] *= daily * weekly;
  }

  // mean-reverting in log-space
  const reversionStrength = 0.001;
  const volatility = 0.03;
  const phi = 1.0 - reversionStrength;
  const series = new Float64Array(nMinutes);
  for (let t = 1; t < nMinutes; t++) {
    const innovation = randnWith(random) * volatility;
    series[t] = phi * series[t - 1] + innovation;
  }
  for (let i = 0; i < nMinutes; i++) ls[i] *= Math.exp(series[i]);

  // per-second lambda and Poisson events per second
  const lsSeconds = repeatEach(ls, 60);
  const nSeconds = lsSeconds.length;
  const rsSeconds = new Float64Array(nSeconds);
  for (let i = 0; i < nSeconds; i++) rsSeconds[i] = poissonWith(lsSeconds[i] / 60, random);

  // per-minute requests
  const rsPerMinute = reshapeSumPerMinute(rsSeconds);

  return {
    xsMinutes,
    rsSeconds,
  }
}

let demandData = null;

function simulate({xsMinutes, rsSeconds, executionTime = 10, keepaliveTime = 60, coldStartTime = 60 }) {
  // Compute total number of busy containers
  const nBusySeconds = rollingSum(rsSeconds, executionTime);

  // Compute total number of draining containers
  const nBusyAndDrainingSeconds = rollingMax(nBusySeconds, keepaliveTime);
  const nDrainingSeconds = new Float64Array(nBusySeconds.length);
  for (let i = 0; i < nDrainingSeconds.length; i++) nDrainingSeconds[i] = nBusyAndDrainingSeconds[i] - nBusySeconds[i];

  // Compute total number of containers that need to start each second
  const nColdStartingRightNowSeconds = new Float64Array(nBusySeconds.length);
  for (let i = 1; i < nColdStartingRightNowSeconds.length; i++) nColdStartingRightNowSeconds[i] = Math.max(0, nBusyAndDrainingSeconds[i] - nBusyAndDrainingSeconds[i - 1]);

  // Compute the total number of cold starting container
  // This ignores the buffer
  const nColdStartingIgnoringBufferSeconds = rollingMax(nColdStartingRightNowSeconds, coldStartTime);

  // percentile 99 of cold starting
  const sorted = Array.from(nColdStartingIgnoringBufferSeconds).sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor(0.99 * sorted.length));
  const nBufferContainers = sorted[idx];
  const nColdStartingSeconds = new Float64Array(nColdStartingIgnoringBufferSeconds.length);
  for (let i = 0; i < nColdStartingSeconds.length; i++) nColdStartingSeconds[i] = Math.min(nColdStartingIgnoringBufferSeconds[i], nBufferContainers);

  // buffer and total seconds series
  const nBufferSeconds = new Float64Array(nBusySeconds.length);
  for (let i = 0; i < nBufferSeconds.length; i++) nBufferSeconds[i] = nBufferContainers - nColdStartingSeconds[i];
  const nTotalSeconds = new Float64Array(nBusySeconds.length);
  for (let i = 0; i < nTotalSeconds.length; i++) nTotalSeconds[i] = nBusySeconds[i] + nDrainingSeconds[i] + nColdStartingSeconds[i] + nBufferSeconds[i];

  // roll up by minutes (mean)
  const nBusyPerMinute = reshapeMeanPerMinute(nBusySeconds);
  const nDrainingPerMinute = reshapeMeanPerMinute(nDrainingSeconds);
  const nColdStartingPerMinute = reshapeMeanPerMinute(nColdStartingSeconds);
  const nBufferPerMinute = reshapeMeanPerMinute(nBufferSeconds);
  const nTotalPerMinute = reshapeMeanPerMinute(nTotalSeconds);

  // build minute-level stacked data
  const timeseries = xsMinutes.map((d, i) => ({
    date: d,
    busy: nBusyPerMinute[i],
    draining: nDrainingPerMinute[i],
    cold: nColdStartingPerMinute[i],
    buffer: nBufferPerMinute[i],
    total: nTotalPerMinute[i],
  }));

  return {
    nBufferContainers,
    timeseries,
  };
}

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

// Log-scale slider mapping (seconds in [1, 3600]) using linear slider [0,1000]
const LOG_MIN = Math.log(1);
const LOG_MAX = Math.log(3600);
const SLIDER_MAX = 1000;
function sliderToSeconds(v) {
  const t = Number(v) / SLIDER_MAX; // 0..1
  const logVal = LOG_MIN + t * (LOG_MAX - LOG_MIN);
  return Math.round(Math.exp(logVal));
}
function secondsToSlider(sec) {
  const clamped = Math.max(1, Math.min(3600, Number(sec)));
  const t = (Math.log(clamped) - LOG_MIN) / (LOG_MAX - LOG_MIN);
  return Math.round(t * SLIDER_MAX);
}

function formatSecondsHuman(totalSeconds) {
  const s = Math.max(1, Math.round(totalSeconds));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (s < 600) return `${m} min ${rem} s`;
  return `${m} min`;
}

// Cost per container per hour (USD)
const CONTAINER_COST = 3.95;

// D3 chart helpers
function clearChart(sel) {
  d3.select(sel).selectAll("*").remove();
}


function stackedAreaChart({ sel, timeseries, x, series, colors, yLabel, domainY }) {
  clearChart(sel);
  const container = d3.select(sel);
  const { width, height } = container.node().getBoundingClientRect();
  const margin = { top: 24, right: 20, bottom: 36, left: 44 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg").attr("width", width).attr("height", height);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleTime().domain(d3.extent(timeseries, x)).range([0, w]);
  const stack = d3.stack().keys(series.map(s => s.key));
  const stackedData = stack(timeseries);
  const maxY = d3.max(timeseries, d => d.busy) * 2;
  const yScale = d3.scaleLinear().domain(domainY ?? [0, maxY]).nice().range([h, 0]);

  const area = d3.area()
    .x(d => xScale(x(d.data)))
    .y0(d => yScale(d[0]))
    .y1(d => yScale(d[1]));

  const xTickFormatter = d3.timeFormat("%b %-d");
  const xAxis = d3.axisBottom(xScale)
    .ticks(d3.timeDay.every(1))
    .tickFormat(xTickFormatter);
  const yAxis = d3.axisLeft(yScale).ticks(5);

  const gx = g.append("g").attr("transform", `translate(0,${h})`).call(xAxis);
  gx.selectAll("text").attr("fill", "#a9afc3");
  gx.selectAll(".domain, .tick line").attr("stroke", "#a9afc3").attr("opacity", 0.3);
  g.append("g").call(yAxis).append("text")
    .attr("fill", "currentColor").attr("x", 4).attr("y", -8)
    .text(yLabel ?? "");

  series.forEach((s, i) => {
    g.append("path")
      .datum(stackedData[i])
      .attr("fill", colors[i] ?? `hsl(${i * 60}, 70%, 55%)`)
      .attr("opacity", 0.6)
      .attr("d", area);
  });

  // Simple legend
  const legend = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  series.forEach((s, i) => {
    const row = legend.append("g").attr("transform", `translate(${i * 160 + 20},0)`);
    row.append("rect").attr("width", 12).attr("height", 12).attr("fill", colors[i] ?? `hsl(${i * 60}, 70%, 55%)`).attr("opacity", 0.6);
    row.append("text").attr("x", 16).attr("y", 10).attr("fill", "#a9afc3").attr("font-size", 12).text(s.label);
  });
}

function run() {
  const executionTime = sliderToSeconds(document.getElementById('execution-time')?.value ?? secondsToSlider(10));
  const keepaliveTime = sliderToSeconds(document.getElementById('keepalive-time')?.value ?? secondsToSlider(60));
  const coldStartTime = sliderToSeconds(document.getElementById('coldstart-time')?.value ?? secondsToSlider(60));
  const {xsMinutes, rsSeconds} = demandData;
  const simulatedData = simulate({ xsMinutes, rsSeconds, executionTime, keepaliveTime, coldStartTime });
  const { timeseries, nBufferContainers } = simulatedData;

  const series = [
    { key: 'busy', label: 'Busy containers' },
    { key: 'draining', label: 'Draining containers' },
    { key: 'cold', label: 'Cold starting containers' },
    { key: 'buffer', label: 'Buffer containers' },
  ];
  stackedAreaChart({ sel: '#chart-number-containers', timeseries, x: d => d.date, series, colors: ['#61d095','#6ea8fe','#9b8cf0','#ffd166'], yLabel: 'number of containers' });

  // Compute total cost over the selected period
  const totalContainerMinutes = timeseries.reduce((acc, d) => acc + d.total, 0);
  const busyContainerMinutes = timeseries.reduce((acc, d) => acc + d.busy, 0);
  const totalContainerHours = totalContainerMinutes / 60;
  const totalCost = totalContainerHours * CONTAINER_COST;
  const costEl = document.getElementById('total-cost');
  if (costEl) {
    const formatted = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(totalCost);
    costEl.textContent = `Total cost: ${formatted}`;
  }

  // Compute and render utilization rate over the full period
  const utilizationRate = totalContainerMinutes > 0 ? busyContainerMinutes / totalContainerMinutes : 0;
  const utilEl = document.getElementById('utilization-rate');
  if (utilEl) {
    utilEl.textContent = ` · Utilization: ${(utilizationRate * 100).toFixed(1)}%`;
  }

  const bufferEl = document.getElementById('buffer-containers');
  if (bufferEl) {
    bufferEl.textContent = ` · Buffer containers: ${nBufferContainers}`;
  }
}

// Wire slider input -> update value labels and rerun chart
function wireParameterControls() {
  const e = document.getElementById('execution-time');
  const k = document.getElementById('keepalive-time');
  const c = document.getElementById('coldstart-time');
  const ev = document.getElementById('execution-time-value');
  const kv = document.getElementById('keepalive-time-value');
  const cv = document.getElementById('coldstart-time-value');
  // initialize slider knob positions from default seconds
  if (e) e.value = String(secondsToSlider(10));
  if (k) k.value = String(secondsToSlider(60));
  if (c) c.value = String(secondsToSlider(60));
  const updateLabels = () => {
    if (ev && e) ev.textContent = formatSecondsHuman(sliderToSeconds(e.value));
    if (kv && k) kv.textContent = formatSecondsHuman(sliderToSeconds(k.value));
    if (cv && c) cv.textContent = formatSecondsHuman(sliderToSeconds(c.value));
  };
  [e, k, c].forEach(input => {
    if (!input) return;
    input.addEventListener('input', () => { updateLabels(); run(); });
    input.addEventListener('change', () => { updateLabels(); run(); });
  });
  updateLabels();
}

function init() {
  console.log("setting data")
  demandData = generateData({ startDate: new Date('2025-05-01'), endDate: new Date('2025-05-03'), seed: 42 });
  console.log("demandData", demandData);
  wireParameterControls();
  run();
}

// Ensure controls are initialized before first render
window.addEventListener('load', init);


