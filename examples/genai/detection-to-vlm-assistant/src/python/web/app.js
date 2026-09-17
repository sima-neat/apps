const elements = {
  title: document.querySelector("#app-title"),
  connectionPill: document.querySelector("#connection-pill"),
  connectionLabel: document.querySelector("#connection-label"),
  phaseKicker: document.querySelector("#phase-kicker"),
  phaseDetail: document.querySelector("#phase-detail"),
  components: document.querySelector("#component-status"),
  video: document.querySelector("#webcam-video"),
  viewerFrame: document.querySelector("#viewer-frame"),
  overlay: document.querySelector("#detection-overlay"),
  viewerPlaceholder: document.querySelector("#viewer-placeholder"),
  cameraMessage: document.querySelector("#camera-message"),
  cameraButton: document.querySelector("#camera-button"),
  restartCamera: document.querySelector("#reload-video"),
  fullscreenVideo: document.querySelector("#fullscreen-video"),
  fps: document.querySelector("#metric-fps"),
  cameraRate: document.querySelector("#camera-rate"),
  currentDetections: document.querySelector("#metric-current"),
  totalDetections: document.querySelector("#metric-total"),
  describedTracks: document.querySelector("#metric-described"),
  latency: document.querySelector("#metric-latency"),
  modelName: document.querySelector("#model-name"),
};

const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d", { alpha: false });
const overlayContext = elements.overlay.getContext("2d");

let currentState = null;
let eventSource = null;
let detectionEventSource = null;
let liveDetectionState = null;
let cameraStream = null;
let localCameraFps = 0;
let encodingInFlight = false;
let queuedUpload = null;
let frameSocket = null;
let frameSocketReconnect = null;
let uploadRetryTimer = null;
let cameraGeneration = 0;
let nextCaptureAt = 0;
let captureLoopStarted = false;

function setConnection(state, label) {
  elements.connectionPill.className = `connection-pill ${state}`;
  elements.connectionLabel.textContent = label;
}

function renderComponents(components) {
  elements.components.replaceChildren();
  const labels = { camera: "Camera", detector: "YOLO", mla: "MLA", vlm: "VLM" };
  Object.entries(labels).forEach(([key, label]) => {
    const component = components[key] || { state: "offline", detail: "Unavailable" };
    const item = document.createElement("span");
    item.className = `component-chip ${component.state}`;
    item.textContent = label;
    item.title = component.detail;
    elements.components.append(item);
  });
}

function render(state) {
  currentState = state;
  const { config, metrics, components } = state;
  document.title = config.title;
  elements.title.textContent = config.title;
  elements.modelName.textContent = config.model_name
    .replace("-Autoround-a16w4", "")
    .replace("-GPTQ-a16w4", "");
  elements.modelName.title = config.model_name;
  elements.phaseKicker.textContent = state.phase.toUpperCase();
  elements.phaseDetail.textContent = state.phase_detail;
  elements.fps.textContent = metrics.fps ? metrics.fps.toFixed(1) : "0.0";
  const captureRate = localCameraFps ? localCameraFps.toFixed(1) : "—";
  const uploadRate = metrics.camera_fps ? metrics.camera_fps.toFixed(1) : "0.0";
  elements.cameraRate.textContent = `Camera: ${captureRate} FPS · upload: ${uploadRate}`;
  elements.currentDetections.textContent = metrics.current_tracks;
  elements.totalDetections.textContent = `${metrics.tracks_created} tracks this session`;
  elements.describedTracks.textContent = `${metrics.tracks_described}/${metrics.tracks_created}`;
  elements.latency.textContent = `VLM latency: ${metrics.vlm_latency_ms ? `${metrics.vlm_latency_ms} ms` : "—"}`;
  renderComponents(components);
}

function showCameraMessage(message, isError = false) {
  elements.cameraMessage.textContent = message;
  elements.viewerPlaceholder.classList.remove("hidden");
  elements.viewerPlaceholder.classList.toggle("error", isError);
}

function stopCamera() {
  cameraGeneration += 1;
  if (frameSocketReconnect) window.clearTimeout(frameSocketReconnect);
  if (uploadRetryTimer) window.clearTimeout(uploadRetryTimer);
  frameSocketReconnect = null;
  uploadRetryTimer = null;
  if (frameSocket) frameSocket.close(1000, "camera stopped");
  frameSocket = null;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  localCameraFps = 0;
  encodingInFlight = false;
  queuedUpload = null;
  nextCaptureAt = 0;
  elements.video.srcObject = null;
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraMessage("Camera access requires this page to be opened over HTTPS", true);
    elements.cameraButton.textContent = "Camera unavailable";
    return;
  }

  showCameraMessage("Requesting webcam permission…");
  stopCamera();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    elements.video.srcObject = cameraStream;
    await elements.video.play();
    connectFrameSocket();
    localCameraFps = Number(cameraStream.getVideoTracks()[0]?.getSettings().frameRate || 0);
    elements.viewerPlaceholder.classList.add("hidden");
    elements.cameraButton.textContent = "Camera active";
    if (!captureLoopStarted) {
      captureLoopStarted = true;
      window.requestAnimationFrame(captureFrame);
    }
  } catch (error) {
    showCameraMessage(`Camera could not start: ${error.message}`, true);
    elements.cameraButton.textContent = "Retry camera";
  }
}

function connectFrameSocket() {
  if (frameSocket && frameSocket.readyState <= WebSocket.OPEN) return;
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/api/frames`);
  socket.binaryType = "arraybuffer";
  frameSocket = socket;
  socket.onopen = () => {
    if (frameSocket !== socket) return;
    setConnection("online", "Live camera connected");
    pumpUploads();
  };
  socket.onerror = () => {
    if (frameSocket === socket) setConnection("connecting", "Camera reconnecting");
  };
  socket.onclose = () => {
    if (frameSocket !== socket) return;
    frameSocket = null;
    if (cameraStream) {
      frameSocketReconnect = window.setTimeout(connectFrameSocket, 500);
    }
  };
}

function pumpUploads() {
  if (!queuedUpload) return;
  if (!frameSocket || frameSocket.readyState !== WebSocket.OPEN) {
    connectFrameSocket();
    return;
  }
  const maxBufferedBytes = Math.max(256 * 1024, queuedUpload.blob.size * 2);
  if (frameSocket.bufferedAmount > maxBufferedBytes) {
    if (!uploadRetryTimer) {
      uploadRetryTimer = window.setTimeout(() => {
        uploadRetryTimer = null;
        pumpUploads();
      }, 5);
    }
    return;
  }
  const pending = queuedUpload;
  queuedUpload = null;
  if (pending.generation !== cameraGeneration) return;
  frameSocket.send(pending.blob);
}

function captureFrame(now) {
  window.requestAnimationFrame(captureFrame);
  if (!cameraStream || encodingInFlight || elements.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  const targetFps = Math.max(1, currentState?.config?.target_fps || 30);
  const frameInterval = 1000 / targetFps;
  if (!nextCaptureAt) nextCaptureAt = now;
  if (now < nextCaptureAt - 1) return;
  nextCaptureAt += frameInterval;
  if (nextCaptureAt < now - frameInterval) nextCaptureAt = now + frameInterval;

  const sourceWidth = elements.video.videoWidth;
  const sourceHeight = elements.video.videoHeight;
  if (!sourceWidth || !sourceHeight) return;
  const maxWidth = Math.max(160, currentState?.config?.upload_max_width || 640);
  const scale = Math.min(1, maxWidth / sourceWidth);
  const width = Math.max(2, Math.round((sourceWidth * scale) / 2) * 2);
  const height = Math.max(2, Math.round((sourceHeight * scale) / 2) * 2);
  if (captureCanvas.width !== width || captureCanvas.height !== height) {
    captureCanvas.width = width;
    captureCanvas.height = height;
  }
  captureContext.drawImage(elements.video, 0, 0, width, height);
  encodingInFlight = true;
  const generation = cameraGeneration;
  captureCanvas.toBlob((blob) => {
    if (generation !== cameraGeneration) return;
    encodingInFlight = false;
    if (!blob) return;
    queuedUpload = { blob, generation };
    pumpUploads();
  }, "image/jpeg", 0.78);
}

function wrapOverlayText(value, maxWidth) {
  const words = String(value || "").split(/\s+/).filter(Boolean);
  if (!words.length) return [""];
  const lines = [];
  let line = words[0];
  words.slice(1).forEach((word) => {
    const candidate = `${line} ${word}`;
    if (overlayContext.measureText(candidate).width <= maxWidth) {
      line = candidate;
    } else {
      lines.push(line);
      line = word;
    }
  });
  lines.push(line);
  return lines;
}

function drawOverlay() {
  window.requestAnimationFrame(drawOverlay);
  const rect = elements.video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(rect.width * dpr));
  const pixelHeight = Math.max(1, Math.round(rect.height * dpr));
  if (elements.overlay.width !== pixelWidth || elements.overlay.height !== pixelHeight) {
    elements.overlay.width = pixelWidth;
    elements.overlay.height = pixelHeight;
  }
  overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayContext.clearRect(0, 0, rect.width, rect.height);

  const source = liveDetectionState?.source || currentState?.source;
  const detections = liveDetectionState?.detections || currentState?.detections || [];
  if (!source?.width || !source?.height || !detections.length) return;
  const scale = Math.min(rect.width / source.width, rect.height / source.height);
  const offsetX = (rect.width - source.width * scale) / 2;
  const offsetY = (rect.height - source.height * scale) / 2;
  overlayContext.lineWidth = 3;
  overlayContext.font = "700 24px Inter, system-ui, sans-serif";

  detections.forEach((detection) => {
    const [x, y, width, height] = detection.bbox;
    const left = offsetX + x * scale;
    const top = offsetY + y * scale;
    const boxWidth = width * scale;
    const boxHeight = height * scale;
    const label = `Person #${detection.track_id} · ${Math.round(detection.score * 100)}%`;
    const trackHue = (Number(detection.track_id) * 137.508) % 360;
    const trackColor = `hsl(${trackHue} 82% 62%)`;
    const trackTextColor = `hsl(${trackHue} 90% 86%)`;
    const trackBackground = `hsl(${trackHue} 45% 14% / .9)`;
    const semantic = detection.semantic_status === "ready"
      ? detection.semantic_description
      : detection.semantic_status === "unavailable"
        ? "description unavailable"
        : detection.semantic_status === "waiting"
          ? "waiting for clear view…"
          : "describing…";
    const labelHeight = 42;
    const textWidth = Math.min(rect.width, overlayContext.measureText(label).width + 24);
    const semanticMaxWidth = Math.min(
      Math.max(320, rect.width * 0.6),
      Math.max(1, rect.width - 16),
    );
    const semanticLines = wrapOverlayText(semantic, semanticMaxWidth - 24);
    const semanticWidth = Math.min(
      rect.width,
      Math.max(...semanticLines.map((line) => overlayContext.measureText(line).width)) + 24,
    );
    const semanticLineHeight = 32;
    const semanticHeight = semanticLines.length * semanticLineHeight + 10;
    const topLabelX = Math.max(0, Math.min(rect.width - textWidth, left));
    const semanticX = Math.max(0, Math.min(rect.width - semanticWidth, left));
    const topLabelY = Math.max(0, top - labelHeight - 2);
    const semanticY = Math.max(0, Math.min(rect.height - semanticHeight, top + boxHeight + 3));
    overlayContext.strokeStyle = trackColor;
    overlayContext.fillStyle = trackBackground;
    overlayContext.strokeRect(left, top, boxWidth, boxHeight);
    overlayContext.fillRect(topLabelX, topLabelY, textWidth, labelHeight);
    overlayContext.fillStyle = trackTextColor;
    overlayContext.fillText(label, topLabelX + 12, topLabelY + 30);
    overlayContext.fillStyle = detection.semantic_status === "ready"
      ? "rgba(28, 52, 43, .94)" : "rgba(43, 38, 21, .92)";
    overlayContext.fillRect(semanticX, semanticY, semanticWidth, semanticHeight);
    overlayContext.fillStyle = detection.semantic_status === "ready" ? trackTextColor : "#ffd19c";
    semanticLines.forEach((line, index) => {
      overlayContext.fillText(
        line,
        semanticX + 12,
        semanticY + 30 + index * semanticLineHeight,
      );
    });
  });
}

async function fetchState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    setConnection("online", "Pipeline connected");
  } catch (_) {
    setConnection("offline", "Reconnecting");
  }
}

function connectEvents() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource("/api/events");
  eventSource.onopen = () => setConnection("online", "Pipeline connected");
  eventSource.onmessage = (event) => {
    try { render(JSON.parse(event.data)); } catch (_) { /* Polling remains as fallback. */ }
  };
  eventSource.onerror = () => setConnection("connecting", "Reconnecting");
}

function connectDetectionEvents() {
  if (detectionEventSource) detectionEventSource.close();
  detectionEventSource = new EventSource("/api/detections");
  detectionEventSource.onmessage = (event) => {
    try {
      liveDetectionState = JSON.parse(event.data);
      elements.currentDetections.textContent = liveDetectionState.detections.length;
    } catch (_) { /* The full state stream remains as fallback. */ }
  };
}

elements.cameraButton.addEventListener("click", startCamera);
elements.restartCamera.addEventListener("click", startCamera);
elements.fullscreenVideo.addEventListener("click", () => {
  if (elements.viewerFrame.requestFullscreen) elements.viewerFrame.requestFullscreen();
});
window.addEventListener("beforeunload", () => {
  if (eventSource) eventSource.close();
  if (detectionEventSource) detectionEventSource.close();
  stopCamera();
});

fetchState();
connectEvents();
connectDetectionEvents();
startCamera();
window.requestAnimationFrame(drawOverlay);
window.setInterval(fetchState, 4000);
