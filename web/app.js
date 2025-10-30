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

function rollingExtrema(xs, window, cmp) {
  // General sliding window extrema via deque (O(n))
  const n = xs.length;
  const out = new Float64Array(n);
  const deq = []; // store indices
  // prefill with window-1 zeros on left
  const padded = padZeroLeft(xs, window - 1);
  for (let i = 0; i < padded.length; i++) {
    const val = padded[i];
    while (deq.length) {
      const backVal = padded[deq[deq.length - 1]];
      // cmp is a comparison function: (a, b) => a <= b for max, a >= b for min
      if (cmp(backVal, val)) deq.pop(); else break;
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

function rollingMax(xs, window) {
  // Largest value in window
  return rollingExtrema(xs, window, (a, b) => a <= b);
}

function rollingMin(xs, window) {
  // Smallest value in window
  return rollingExtrema(xs, window, (a, b) => a >= b);
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

function reshapeSumPerHour(xsSeconds) {
  const hours = xsSeconds.length / 3600;
  const out = new Float64Array(hours);
  for (let h = 0; h < hours; h++) {
    let s = 0;
    for (let j = 0; j < 3600; j++) s += xsSeconds[h * 3600 + j];
    out[h] = s;
  }
  return out;
}

function generateData({ startDate, endDate, seed, baseRate = 100 }) {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const nMinutes = Math.floor((end - start) / 60000);
  const xsByMinute = dateRangeMinutes(start, end);
  const js = range(nMinutes);
  const random = makeRng(seed);

  // base rate with daily and weekly sin components
  let lsByMinute = new Float64Array(nMinutes).fill(baseRate);
  for (let i = 0; i < nMinutes; i++) {
    const daily = Math.sin((js[i] / (60 * 24)) * 2 * Math.PI) * 0.67 + 1;
    const weekly = Math.sin((js[i] / (60 * 24 * 7)) * 2 * Math.PI) * 0.2 + 1;
    lsByMinute[i] *= daily * weekly;
  }

  // mean-reverting in log-space
  const reversionStrength = 0.001;
  const volatility = 0.03;
  const phi = 1.0 - reversionStrength;
  const seriesByMinute = new Float64Array(nMinutes);
  for (let t = 1; t < nMinutes; t++) {
    const innovation = randnWith(random) * volatility;
    seriesByMinute[t] = phi * seriesByMinute[t - 1] + innovation;
  }
  for (let i = 0; i < nMinutes; i++) lsByMinute[i] *= Math.exp(seriesByMinute[i]);

  // per-second lambda and Poisson events per second
  const lsBySecond = repeatEach(lsByMinute, 60);
  const nSeconds = lsBySecond.length;
  const rsBySecond = new Float64Array(nSeconds);
  for (let i = 0; i < nSeconds; i++) rsBySecond[i] = poissonWith(lsBySecond[i] / 60, random);

  return {
    xsByMinute,
    rsBySecond,
  }
}

let demandData = null;

function simulate({xsByMinute, rsBySecond, executionTime = 10, keepaliveTime = 60, coldStartTime = 60, nBufferContainers = 0, nWarmContainers = 0 }) {
  const nSeconds = rsBySecond.length;

  // initialize empty arrays
  const nBusyBySecond = new Float64Array(nSeconds);
  const nIdleBySecond = new Float64Array(nSeconds);
  const nColdStartingBySecond = new Float64Array(nSeconds);
  const nTotalBySecond = new Float64Array(nSeconds);
  const totalQueueTimeBySecond = new Float64Array(nSeconds);

  // Every container is in one of three states: cold starting, busy, idle
  const coldStartingContainers = [];  // By what time they started
  const busyContainers = [];  // By what time they started
  const idleContainers = [];  // By what time they started
  const requests = [];  // Queue of requests

  // For all queues, we don't pop the front (due to JS array limitations), just use an index
  let coldStartingJ = 0;
  let busyJ = 0;
  let idleJ = 0;
  let requestsJ = 0;

  // Step through and simulate each second
  for (let i = 0; i < nSeconds; i++) {
    // Cold starting containers -> idle
    while (coldStartingContainers.length > 0 && coldStartingContainers[coldStartingJ] + coldStartTime <= i) {
      idleContainers.push(i);
      coldStartingJ++;
    }
    // Finish busy containers
    while (busyContainers.length > busyJ && busyContainers[busyJ] + executionTime < i) {
      idleContainers.push(i);
      busyJ++;
    }
    // Add new requests to queue
    for (let z = 0; z < rsBySecond[i]; z++) {
      requests.push(i);
    }
    // Assign requests to idle containers
    while (requests.length > requestsJ && idleContainers.length > idleJ) {
      const requestI = requests[requestsJ++];
      busyContainers.push(requestI);
      totalQueueTimeBySecond[requestI] += i - requestI;
      idleContainers.pop(); // Remove the last idle container
    }
    // Compute container counts
    const nIdleContainers = idleContainers.length - idleJ;
    const nBusyContainers = busyContainers.length - busyJ;
    const nColdStartingContainers = coldStartingContainers.length - coldStartingJ;
    const nTotalContainers = nIdleContainers + nBusyContainers + nColdStartingContainers;

    // Write stats to arrays
    nIdleBySecond[i] = nIdleContainers;
    nBusyBySecond[i] = nBusyContainers;
    nColdStartingBySecond[i] = nColdStartingContainers;
    nTotalBySecond[i] = nTotalContainers;

    // Start new containers based on queue size
    const queueSize = requests.length - requestsJ;
    const nDesiredTotalContainers = Math.max(nWarmContainers + nColdStartingContainers, nBusyContainers + queueSize + nBufferContainers);
    const nDesiredNewContainers = nDesiredTotalContainers - nTotalContainers;
    let nDesiredShutdownContainers = -nDesiredNewContainers;
    for (let z = 0; z < nDesiredNewContainers; z++) {
      coldStartingContainers.push(i);
    }

    // Shut down idle containers if we're above the desired total
    while (idleContainers.length > idleJ && idleContainers[idleJ] + keepaliveTime < i && nDesiredShutdownContainers > 0) {
      idleJ++;
      nDesiredShutdownContainers--;
    }
  }

  // roll up by minutes (mean)
  const nBusyPerMinute = reshapeMeanPerMinute(nBusyBySecond);
  const nIdlePerMinute = reshapeMeanPerMinute(nIdleBySecond);
  const nColdStartingPerMinute = reshapeMeanPerMinute(nColdStartingBySecond);
  const nTotalPerMinute = reshapeMeanPerMinute(nTotalBySecond);
  const totalQueueTimePerHour = reshapeSumPerHour(totalQueueTimeBySecond);
  const totalRequestsPerHour = reshapeSumPerHour(rsBySecond);

  // build minute-level stacked data of containers.
  // Also trim the first 24 hours of data, which we treat as "warmup"
  const containers = xsByMinute.map((d, i) => ({
    date: d,
    busy: nBusyPerMinute[i],
    draining: nIdlePerMinute[i],
    cold: nColdStartingPerMinute[i],
    total: nTotalPerMinute[i],
  })).slice(24 * 60);

  // build hour-level stacked data of queue stats
  // Also trim the first 24 hours of data
  const xsByHour = [];
  for (let i = 0; i < xsByMinute.length; i += 60) {
    xsByHour.push(xsByMinute[i]);
  }
  const queueStats = xsByHour.map((d, i) => ({
    date: d,
    queueTime: totalQueueTimePerHour[i],
    requests: totalRequestsPerHour[i],
  })).slice(24);

  return { containers, queueStats };
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

// Log-scale slider mapping for requests per minute (rpm in [1, 10000])
const LOG_RPM_MIN = Math.log(1);
const LOG_RPM_MAX = Math.log(10000);
function sliderToRpm(v) {
  const t = Number(v) / SLIDER_MAX;
  const logVal = LOG_RPM_MIN + t * (LOG_RPM_MAX - LOG_RPM_MIN);
  return Math.max(1, Math.round(Math.exp(logVal)));
}
function rpmToSlider(rpm) {
  const clamped = Math.max(1, Math.min(10000, Number(rpm)));
  const t = (Math.log(clamped) - LOG_RPM_MIN) / (LOG_RPM_MAX - LOG_RPM_MIN);
  return Math.round(t * SLIDER_MAX);
}
function formatRpmHuman(rpm) {
  const r = Math.round(rpm);
  if (r >= 1000) return `${(r / 1000).toFixed(r % 1000 === 0 ? 0 : 1)}k`;
  return String(r);
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

function stackedAreaWithRightLine({ sel, containers, x, series, colors, yLabelLeft, domainYLeft, lineData, lineY, yLabelRight, domainYRight }) {
  clearChart(sel);
  const container = d3.select(sel);
  const { width, height } = container.node().getBoundingClientRect();
  const margin = { top: 24, right: 44, bottom: 36, left: 44 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg").attr("width", width).attr("height", height);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleTime().domain(d3.extent(containers, x)).range([0, w]);
  const stack = d3.stack().keys(series.map(s => s.key));
  const stackedData = stack(containers);
  const maxYLeft = d3.max(containers, d => d.busy) * 2;
  const yScaleLeft = d3.scaleLinear().domain(domainYLeft ?? [0, maxYLeft]).nice().range([h, 0]);

  // const maxRight = lineData && lineData.length ? (d3.max(lineData, lineY) ?? 0) : 0;
  const maxRight = 5;  // Use a fixed max so it doesn't rescale
  const yScaleRight = d3.scaleLinear().domain(domainYRight ?? [0, maxRight]).nice().range([h, 0]);

  const area = d3.area()
    .x(d => xScale(x(d.data)))
    .y0(d => yScaleLeft(d[0]))
    .y1(d => yScaleLeft(d[1]));

  const xTickFormatter = d3.timeFormat("%b %-d");
  const xAxis = d3.axisBottom(xScale)
    .ticks(d3.timeDay.every(1))
    .tickFormat(xTickFormatter);
  const yAxisLeft = d3.axisLeft(yScaleLeft).ticks(5);
  const yAxisRight = d3.axisRight(yScaleRight).ticks(5);

  // Gridlines based on left axis
  const xGrid = d3.axisBottom(xScale)
    .ticks(d3.timeDay.every(1))
    .tickSize(-h)
    .tickFormat("");
  const yGrid = d3.axisLeft(yScaleLeft)
    .ticks(5)
    .tickSize(-w)
    .tickFormat("");
  g.append("g")
    .attr("class", "grid grid-x")
    .attr("transform", `translate(0,${h})`)
    .call(xGrid)
    .selectAll(".tick line").attr("stroke", "#a9afc3").attr("opacity", 0.15);
  g.selectAll(".grid.grid-x .domain").remove();
  g.append("g")
    .attr("class", "grid grid-y")
    .call(yGrid)
    .selectAll(".tick line").attr("stroke", "#a9afc3").attr("opacity", 0.15);
  g.selectAll(".grid.grid-y .domain").remove();

  const gx = g.append("g").attr("transform", `translate(0,${h})`).call(xAxis);
  gx.selectAll("text").attr("fill", "#a9afc3");
  gx.selectAll(".domain, .tick line").attr("stroke", "#a9afc3").attr("opacity", 0.3);
  g.append("g").call(yAxisLeft).append("text")
    .attr("fill", "currentColor").attr("x", 4).attr("y", -8)
    .text(yLabelLeft ?? "");
  const gyRight = g.append("g").attr("transform", `translate(${w},0)`).call(yAxisRight);
  gyRight.append("text")
    .attr("fill", "currentColor").attr("x", -4).attr("y", -8)
    .attr("text-anchor", "end")
    .text(yLabelRight ?? "");

  series.forEach((s, i) => {
    g.append("path")
      .datum(stackedData[i])
      .attr("fill", colors[i] ?? `hsl(${i * 60}, 70%, 55%)`)
      .attr("opacity", 0.6)
      .attr("d", area);
  });

  if (lineData && lineData.length) {
    console.log(lineData);
    const line = d3.line()
      .x(d => xScale(x(d)))
      .y(d => yScaleRight(lineY(d)));
    g.append("path")
      .datum(lineData)
      .attr("fill", "none")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "6,4")
      .attr("d", line);
  }

  // Legend: stacked series + line
  const legend = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  series.forEach((s, i) => {
    const row = legend.append("g").attr("transform", `translate(${i * 160 + 20},0)`);
    row.append("rect").attr("width", 12).attr("height", 12).attr("fill", colors[i] ?? `hsl(${i * 60}, 70%, 55%)`).attr("opacity", 0.6);
    row.append("text").attr("x", 16).attr("y", 10).attr("fill", "#a9afc3").attr("font-size", 12).text(s.label);
  });
  const lineRow = legend.append("g").attr("transform", `translate(${series.length * 160 + 20},0)`);
  lineRow.append("line").attr("x1", 0).attr("x2", 24).attr("y1", 6).attr("y2", 6).attr("stroke", "#fff").attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
  lineRow.append("text").attr("x", 28).attr("y", 10).attr("fill", "#a9afc3").attr("font-size", 12).text("Avg queue time");
}

function run() {
  const executionTime = sliderToSeconds(document.getElementById('execution-time')?.value ?? secondsToSlider(10));
  const keepaliveTime = sliderToSeconds(document.getElementById('keepalive-time')?.value ?? secondsToSlider(60));
  const coldStartTime = sliderToSeconds(document.getElementById('coldstart-time')?.value ?? secondsToSlider(60));
  const nBufferContainers = Number(document.getElementById('buffer-containers')?.value ?? 0);
  const nWarmContainers = Number(document.getElementById('warm-containers')?.value ?? 0);
  const {xsByMinute, rsBySecond} = demandData;
  const simulatedData = simulate({ xsByMinute, rsBySecond, executionTime, keepaliveTime, coldStartTime, nBufferContainers, nWarmContainers });
  const { containers, queueStats } = simulatedData;

  const series = [
    { key: 'busy', label: 'Busy containers' },
    { key: 'cold', label: 'Cold starting containers' },
    { key: 'draining', label: 'Draining containers' },
  ];
  const colors = ['#bef264','#fde047','#6e47fd'];
  // Average queue time per request (skip minutes with zero requests)
  const qprData = queueStats
    .filter(t => t.requests > 0)
    .map(t => ({ date: t.date, value: t.queueTime / t.requests }));

  stackedAreaWithRightLine({
    sel: '#chart-number-containers',
    containers,
    x: d => d.date,
    series,
    colors: colors,
    yLabelLeft: 'number of containers',
    lineData: qprData,
    lineY: d => d.value,
    yLabelRight: 'avg queue time (s)'
  });

  // Compute total cost over the selected period
  const totalContainerMinutes = containers.reduce((acc, d) => acc + d.total, 0);
  const busyContainerMinutes = containers.reduce((acc, d) => acc + d.busy, 0);
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
}

// Wire slider input -> update value labels and rerun chart
function wireParameterControls() {
  const r = document.getElementById('requests-per-minute');
  const e = document.getElementById('execution-time');
  const k = document.getElementById('keepalive-time');
  const c = document.getElementById('coldstart-time');
  const b = document.getElementById('buffer-containers');
  const w = document.getElementById('warm-containers');
  const rv = document.getElementById('requests-per-minute-value');
  const ev = document.getElementById('execution-time-value');
  const kv = document.getElementById('keepalive-time-value');
  const cv = document.getElementById('coldstart-time-value');
  const bv = document.getElementById('buffer-containers-value');
  const wv = document.getElementById('warm-containers-value');
  // initialize slider knob positions from defaults
  if (r) r.value = String(rpmToSlider(100));
  if (e) e.value = String(secondsToSlider(10));
  if (k) k.value = String(secondsToSlider(60));
  if (c) c.value = String(secondsToSlider(60));
  if (b) b.value = String(0);
  if (w) w.value = String(0);
  const updateLabels = () => {
    if (rv && r) rv.textContent = formatRpmHuman(sliderToRpm(r.value));
    if (ev && e) ev.textContent = formatSecondsHuman(sliderToSeconds(e.value));
    if (kv && k) kv.textContent = formatSecondsHuman(sliderToSeconds(k.value));
    if (cv && c) cv.textContent = formatSecondsHuman(sliderToSeconds(c.value));
    if (bv && b) bv.textContent = String(Math.round(b.value));
    if (wv && w) wv.textContent = String(Math.round(w.value));
  };
  const regenerateDataFromControls = () => {
    const baseRate = sliderToRpm(r?.value ?? rpmToSlider(100));
    demandData = generateData({ startDate: DEFAULT_START, endDate: DEFAULT_END, seed: DEFAULT_SEED, baseRate });
  };
  [e, k, c, b, w].forEach(input => {
    if (!input) return;
    input.addEventListener('input', () => { updateLabels(); run(); });
    input.addEventListener('change', () => { updateLabels(); run(); });
  });
  if (r) {
    r.addEventListener('input', () => { updateLabels(); regenerateDataFromControls(); run(); });
    r.addEventListener('change', () => { updateLabels(); regenerateDataFromControls(); run(); });
  }
  updateLabels();
}

const DEFAULT_START = new Date('2025-05-01');
const DEFAULT_END = new Date('2025-05-08');
const DEFAULT_SEED = 42;
let CURRENT_SEED = DEFAULT_SEED;

function regenerateWithSeed(seed) {
  const r = document.getElementById('requests-per-minute');
  const baseRate = sliderToRpm(r?.value ?? rpmToSlider(100));
  demandData = generateData({ startDate: DEFAULT_START, endDate: DEFAULT_END, seed, baseRate });
  run();
}

function init() {
  wireParameterControls();
  // Initial demand generation based on current slider state
  CURRENT_SEED = DEFAULT_SEED;
  regenerateWithSeed(CURRENT_SEED);
  const regenBtn = document.getElementById('regenerate');
  if (regenBtn) {
    regenBtn.addEventListener('click', () => {
      CURRENT_SEED = Math.floor(Math.random() * 0xFFFFFFFF);
      regenerateWithSeed(CURRENT_SEED);
    });
  }
}

// Ensure controls are initialized before first render
window.addEventListener('load', init);


