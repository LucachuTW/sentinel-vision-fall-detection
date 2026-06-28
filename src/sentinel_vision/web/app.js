const $ = (id) => document.getElementById(id);
let lastState = null;

// When the API requires auth, open the dashboard as /?token=… ; it is forwarded to every call.
const API_TOKEN = new URLSearchParams(location.search).get("token");
const withToken = (path) =>
  API_TOKEN ? `${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(API_TOKEN)}` : path;

function formatTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleTimeString([], {hour12: false});
}

function renderState(state) {
  lastState = state;
  $("sourceName").textContent = state.source_id || "Unknown source";
  $("sequence").textContent = `FRAME ${String(state.sequence || 0).padStart(6, "0")}`;
  $("fps").textContent = Number(state.fps || 0).toFixed(1);
  $("e2e").textContent = Math.round(state.end_to_end_ms || 0);
  $("trackCount").textContent = (state.tracks || []).length;
  $("eventCount").textContent = state.events_total || 0;
  $("updated").textContent = formatTime(state.captured_at);

  const latency = Object.entries(state.latency_ms || {});
  $("latencyStrip").innerHTML = latency.length
    ? latency.map(([name, value]) => `<span>${name}<b>${Number(value).toFixed(1)} ms</b></span>`).join("")
    : "<span>Waiting for telemetry…</span>";

  const actionMap = new Map((state.actions || []).map((item) => [item.track_id, item]));
  const tracks = state.tracks || [];
  $("tracks").classList.toggle("empty", !tracks.length);
  $("tracks").innerHTML = tracks.length ? tracks.map((track) => {
    const action = actionMap.get(track.track_id) || {label: "warming up", confidence: 0};
    return `<div class="track">
      <span class="track-id">${track.track_id}</span>
      <div><b>${action.label}</b><small>pose ${(track.keypoint_confidence * 100).toFixed(0)}% · ${track.label}</small></div>
      <span class="confidence">${(action.confidence * 100).toFixed(0)}%</span>
    </div>`;
  }).join("") : "No active tracks";
}

async function refreshHealth() {
  try {
    const response = await fetch("/health/ready", {cache: "no-store"});
    const health = await response.json();
    const status = $("status");
    status.className = `status ${health.ready ? "ready" : health.status === "error" ? "error" : ""}`;
    status.innerHTML = `<i></i> ${health.ready ? "Pipeline ready" : health.status}`;
    $("environment").textContent = health.environment.toUpperCase();
    $("components").innerHTML = Object.entries(health.components || {}).map(([key, value]) =>
      `<div title="${value}"><dt>${key}</dt><dd>${value}</dd></div>`).join("");
    const warnings = health.degradations || [];
    $("degradations").hidden = !warnings.length;
    $("degradations").innerHTML = warnings.length
      ? `<b>DEMO DEGRADATIONS</b><br>${warnings.map((item) => `• ${item}`).join("<br>")}` : "";
  } catch (_) {
    $("status").className = "status error";
    $("status").innerHTML = "<i></i> API unavailable";
  }
}

async function refreshSystemMetrics() {
  try {
    const response = await fetch(withToken("/v1/system"), {cache: "no-store"});
    const system = await response.json();
    const gpu = system.gpu;
    $("gpuUtil").textContent = gpu ? `${Math.round(gpu.utilization_ratio * 100)}%` : "—";
    $("gpuVram").textContent = gpu ? `${(gpu.memory_used_bytes / 1073741824).toFixed(1)}G` : "—";
  } catch (_) { /* optional GPU telemetry */ }
}

function plotSeries(context, points, xAt, yAt, color) {
  if (!points.length) return;
  context.beginPath();
  context.strokeStyle = color;
  context.lineWidth = 2;
  points.forEach((point, index) => {
    const x = xAt(index);
    const y = yAt(point);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
}

function drawTelemetry(samples) {
  const canvas = $("telemetryChart");
  const empty = $("chartEmpty");
  empty.style.display = samples.length < 2 ? "grid" : "none";
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  if (samples.length < 2) return;

  const pad = {left: 42, right: 42, top: 12, bottom: 25};
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const maxLatency = Math.max(10, ...samples.map((sample) => sample.end_to_end_ms));
  const maxFps = Math.max(1, ...samples.map((sample) => sample.fps));
  context.strokeStyle = "#202c3b";
  context.fillStyle = "#8492a5";
  context.font = "9px IBM Plex Mono, monospace";
  context.lineWidth = 1;
  for (let row = 0; row <= 4; row += 1) {
    const y = pad.top + (plotHeight * row / 4);
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
    context.fillText(`${Math.round(maxLatency * (1 - row / 4))}ms`, 2, y + 3);
    context.fillText(`${Math.round(maxFps * (1 - row / 4))}`, width - pad.right + 8, y + 3);
  }
  const xAt = (index) => pad.left + (plotWidth * index / (samples.length - 1));
  plotSeries(context, samples.map((sample) => sample.end_to_end_ms), xAt,
    (value) => pad.top + plotHeight * (1 - value / maxLatency), "#5dd6e9");
  plotSeries(context, samples.map((sample) => sample.fps), xAt,
    (value) => pad.top + plotHeight * (1 - value / maxFps), "#6ce3a4");
}

async function refreshTelemetry() {
  try {
    const response = await fetch(withToken("/v1/telemetry?limit=120"), {cache: "no-store"});
    drawTelemetry(await response.json());
  } catch (_) { /* transient telemetry polling failure */ }
}

async function refreshEvents() {
  try {
    const response = await fetch(withToken("/v1/events?limit=20"), {cache: "no-store"});
    const events = await response.json();
    if (!events.length) return;
    $("events").innerHTML = events.map((event) => `<tr>
      <td>${formatTime(event.occurred_at)}</td><td>#${event.track_id}</td><td>${event.action}</td>
      <td>${(event.confidence * 100).toFixed(0)}%</td>
      <td><span class="severity ${event.severity}">${event.severity}</span></td>
      <td>${event.narrative || "—"}</td>
    </tr>`).join("");
  } catch (_) { /* transient dashboard polling failure */ }
}

function toggleSourceInputs() {
  const kind = $("srcKind").value;
  $("srcDevice").hidden = kind !== "camera";
  $("srcRtsp").hidden = kind !== "rtsp";
  $("srcFile").hidden = kind !== "file";
}

function reloadStream() {
  const base = withToken("/v1/stream.mjpeg");
  $("stream").src = `${base}${base.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

async function applySource() {
  const kind = $("srcKind").value;
  const status = $("sourceStatus");
  status.textContent = "switching…";
  try {
    let response;
    if (kind === "file") {
      const file = $("srcFile").files[0];
      if (!file) { status.textContent = "pick a file"; return; }
      response = await fetch(withToken(`/v1/source/upload?filename=${encodeURIComponent(file.name)}`),
        {method: "POST", body: file});
    } else {
      const payload = {kind};
      if (kind === "camera") payload.uri = $("srcDevice").value;
      if (kind === "rtsp") payload.uri = $("srcRtsp").value;
      response = await fetch(withToken("/v1/source"),
        {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      status.textContent = `error: ${detail.detail || response.status}`;
      return;
    }
    status.textContent = "switched";
    setTimeout(reloadStream, 600);
  } catch (_) {
    status.textContent = "request failed";
  }
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}${withToken("/v1/ws")}`);
  socket.onmessage = (event) => renderState(JSON.parse(event.data));
  socket.onclose = () => setTimeout(connectSocket, 1500);
  socket.onerror = () => socket.close();
}

if (API_TOKEN) $("stream").src = withToken("/v1/stream.mjpeg");
$("srcKind").addEventListener("change", toggleSourceInputs);
$("srcApply").addEventListener("click", applySource);
toggleSourceInputs();
connectSocket();
refreshHealth();
refreshEvents();
refreshSystemMetrics();
refreshTelemetry();
setInterval(refreshHealth, 3000);
setInterval(refreshEvents, 2500);
setInterval(refreshSystemMetrics, 3000);
setInterval(refreshTelemetry, 2500);
window.addEventListener("resize", refreshTelemetry);
