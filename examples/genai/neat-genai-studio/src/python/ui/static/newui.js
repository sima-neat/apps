// New Dashboard Elements
const cameraPreview = document.getElementById('cameraPreview');
const snapAnimation = document.getElementById('snapAnimation');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const recordButton = document.getElementById('recordButton');
const recordIcon = document.getElementById('recordIcon');
const recordIconUrls = window.recordIconUrls;
const snapChatButton = document.getElementById('snapChatButton');
const uploadButton = document.getElementById('uploadButton');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const systemPromptButton = document.getElementById('systemPromptButton');
const systemPromptModal = document.getElementById('systemPromptModal');
const systemPromptTextarea = document.getElementById('systemPromptTextarea');
const systemPromptSave = document.getElementById('systemPromptSave');
const systemPromptCancel = document.getElementById('systemPromptCancel');
const systemPromptModalMessage = document.getElementById('systemPromptModalMessage');

let isMicrophoneMuted = true;
let mediaStream = null;
let cameraStartPromise = null;
let audioTracks = [];
let recordedChunks = [];

const socket = io('/');
const audioQueue = [];
let isPlaying = false;
let currentAudioContext = null;
let receivedEndSignal = false;
let shouldPlayAudio = true;
let scheduledSources = [];
let scheduledHighlightTimers = [];
let nextStartTime = 0;
let activeGeneration = false;
let pendingNewGenerationAudio = false;
// Bumped every time playback is stopped/aborted. Any in-flight or queued audio
// task captures the epoch it began under and bails the moment it changes, so a
// chunk decoded just before Stop can never start after it.
let audioEpoch = 0;
let requestQueue = Promise.resolve();

// First Audio timing tracking
let userInputStartTime = null;
let firstAudioStarted = false;
// Tokens streamed for the in-flight assistant reply (one per content delta,
// same as the benchmark counts), shown on the message when it finishes.
let currentGenTokens = 0;

let ragServerStatusText = "";

let currentSystemPrompt = '';
let systemPromptRequestInFlight = false;
// Preferred TTS engine (defaults to piper-tts, matching the backend). Drives
// which voice-selection setting is shown (rhasspy voice vs piper-plus voice).
let _currentTtsEngine = 'piper-tts';
let _ppVoicesAvailable = false;
// Used for TTS routing while the transcription selector remains on Auto-detect.
// English is the safe default for typed prompts before the first recording.
let _detectedSpeechLanguage = 'en';

function isLlmOnlyMode() {
  return !selectedChatModelSupportsVision();
}

function isRagEnabled() {
  const config = window.SIMA_CONFIG || {};
  return config.ragEnabled === true || config.ragEnabled === 'true';
}

function getConfiguredChatModels() {
  const config = window.SIMA_CONFIG || {};
  return Array.isArray(config.chatModels) ? config.chatModels.filter(Boolean) : [];
}

function getChatModelCapabilities() {
  const config = window.SIMA_CONFIG || {};
  return config.chatModelCapabilities && typeof config.chatModelCapabilities === 'object'
    ? config.chatModelCapabilities
    : {};
}

// The model actually resident on the accelerator and used for chat + vision.
// The dropdown is only a *selection* that the Load button acts on, so it can
// differ from the active model while the user browses the catalog.
let _activeChatModel = '';
// Match the original Multimodal Assistant behavior: automatically enable image
// prompting whenever a vision-capable model becomes active.
let _visionPromptModel = '';

function getSelectedChatModel() {
  // In control mode the loaded model is authoritative — the dropdown selection
  // only becomes active once the user loads it. Fall back to the dropdown /
  // config default when there is no control API (static preloaded models).
  if (controlEnabled()) return _activeChatModel || '';
  const select = document.getElementById('chatModelSelect');
  if (select) return select.value || '';
  const models = getConfiguredChatModels();
  return window.SIMA_CONFIG?.defaultChatModel || models[0] || '';
}

function selectedChatModelSupportsVision() {
  const model = getSelectedChatModel();
  if (!model) return false;
  const capabilities = getChatModelCapabilities();
  const modelCaps = capabilities[model] || {};
  if (Object.prototype.hasOwnProperty.call(modelCaps, 'supportsVision')) {
    return modelCaps.supportsVision === true;
  }
  // Preserve compatibility with static/legacy launches that identify an LLM
  // using the rendered body class but do not publish per-model capabilities.
  return !document.body.classList.contains('llm-only');
}

function initializeChatModelSelect() {
  const row = document.getElementById('chatModelRow');
  const select = document.getElementById('chatModelSelect');
  const models = getConfiguredChatModels();

  if (!row || !select || models.length === 0) return;

  while (select.firstChild) {
    select.removeChild(select.firstChild);
  }
  models.forEach(model => {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
  });

  select.value = window.SIMA_CONFIG?.defaultChatModel || models[0];
  select.addEventListener('change', updateSelectedModelVisionState);
  row.style.display = models.length > 1 ? 'block' : 'none';
  updateSelectedModelVisionState();
}

function updateSelectedModelVisionState() {
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');
  const activeModel = getSelectedChatModel();
  const supportsVision = selectedChatModelSupportsVision();
  const activatedVisionModel = supportsVision && activeModel !== _visionPromptModel;
  document.body.classList.toggle('llm-only', !supportsVision);

  if (includeImageCheckbox) {
    if (!supportsVision) {
      includeImageCheckbox.checked = false;
      includeImageCheckbox.disabled = true;
      _visionPromptModel = '';
    } else {
      // A newly activated VLM should be immediately multimodal. Repeated catalog
      // refreshes for the same model must not override a later manual opt-out.
      if (activatedVisionModel) includeImageCheckbox.checked = true;
      includeImageCheckbox.disabled = false;
      _visionPromptModel = activeModel;
    }
    toggleImageButtons(includeImageCheckbox.checked);
  }

  if (activatedVisionModel) revealCameraForVisionModel();

  if (!supportsVision) {
    if (mediaStream) {
      mediaStream.getVideoTracks().forEach(track => {
        track.stop();
        mediaStream.removeTrack(track);
      });
    }
    if (activeModel && (!mediaStream || mediaStream.getAudioTracks().length === 0)) {
      startAudioOnly();
    }
  } else if (!mediaStream || mediaStream.getVideoTracks().length === 0) {
    startCamera();
  }
}

function hideRagControlsIfDisabled() {
  if (isRagEnabled()) return;

  // Hide the RAG tab (its panel shows only when active anyway).
  const ragTab = document.getElementById('tabRag');
  const ragCheckbox = document.getElementById('toggleRAG');

  if (ragTab) {
    ragTab.style.display = 'none';
  }
  if (ragCheckbox) {
    ragCheckbox.checked = false;
  }
}

function getVisionImageSize() {
  const config = window.SIMA_CONFIG || {};
  const selectedModel = getSelectedChatModel();
  const modelCaps = getChatModelCapabilities()[selectedModel] || {};
  const modelSize = modelCaps.imageSize || modelCaps.visionImageSize || null;
  const height = parseInt(modelSize?.height ?? config.visionImageHeight, 10);
  const width = parseInt(modelSize?.width ?? config.visionImageWidth, 10);

  if (Number.isFinite(height) && Number.isFinite(width) && height > 0 && width > 0) {
    return { height, width };
  }

  return null;
}

// Access the local camera and microphone feed
async function startCamera() {
  if (cameraStartPromise) {
    return cameraStartPromise;
  }

  cameraStartPromise = startCameraInternal();
  try {
    return await cameraStartPromise;
  } finally {
    cameraStartPromise = null;
  }
}

async function startCameraInternal() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Browser camera access requires HTTPS and media-device support');
    }
    const previousStream = mediaStream;
    const nextStream = await navigator.mediaDevices.getUserMedia({
      video: buildVideoConstraint(),
      audio: buildAudioConstraint()
    });
    if (previousStream && previousStream !== nextStream) {
      previousStream.getTracks().forEach(track => track.stop());
    }
    mediaStream = nextStream;

    cameraPreview.srcObject = mediaStream;
    audioTracks = mediaStream.getAudioTracks();
    toggleMicrophone(true);
    setCameraStatus('Live Camera');

    // Set up dynamic camera container sizing
    setupCameraContainer();
    // Labels are only available after permission is granted.
    refreshDeviceLists();
    return true;
  } catch (error) {
    console.error('Error accessing media devices.', error);
    setCameraStatus(cameraAccessErrorMessage(error), true);
    return false;
  }
}

function cameraAccessErrorMessage(error) {
  if (error && error.name === 'NotAllowedError') return 'Camera blocked — allow access';
  if (error && error.name === 'NotFoundError') return 'No browser camera found';
  if (error && error.name === 'NotReadableError') return 'Camera unavailable or busy';
  return 'Camera unavailable — check HTTPS';
}

function setCameraStatus(message, isError = false) {
  const badge = document.getElementById('cameraStatus');
  if (!badge) return;
  badge.textContent = message;
  badge.classList.toggle('is-error', isError);
  badge.title = isError ? message : '';
}

async function startAudioOnly() {
  try {
    // Only request audio access, no video
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: buildAudioConstraint()
    });

    audioTracks = mediaStream.getAudioTracks();
    toggleMicrophone(true);
    refreshDeviceLists();
    console.log('Audio-only mode initialized for LLM');
  } catch (error) {
    console.error('Error accessing audio devices.', error);
  }
}

// ---- Camera / microphone / speaker device selection ----
let selectedVideoDeviceId = localStorage.getItem('studioCameraId') || '';
let selectedAudioDeviceId = localStorage.getItem('studioMicId') || '';
let selectedSpeakerDeviceId = localStorage.getItem('studioSpeakerId') || '';

// Route an AudioContext's output to the chosen speaker. Chromium 110+ only;
// elsewhere (and for the built-in browser voice) audio stays on the default.
function speakerRoutingSupported() {
  return typeof AudioContext !== 'undefined' && 'setSinkId' in AudioContext.prototype;
}
async function applySpeakerSink(ctx) {
  if (!ctx || !selectedSpeakerDeviceId || typeof ctx.setSinkId !== 'function') return;
  try { await ctx.setSinkId(selectedSpeakerDeviceId); }
  catch (e) { console.warn('Could not route audio to the selected speaker:', e); }
}

function buildVideoConstraint() {
  const c = { width: { ideal: 1920 }, height: { ideal: 1080 } };
  if (selectedVideoDeviceId) c.deviceId = { ideal: selectedVideoDeviceId };
  return c;
}
function buildAudioConstraint() {
  return selectedAudioDeviceId ? { deviceId: { ideal: selectedAudioDeviceId } } : true;
}

async function refreshDeviceLists() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  let devices = [];
  try { devices = await navigator.mediaDevices.enumerateDevices(); } catch (e) { return; }
  populateDeviceSelect('cameraSelect', devices.filter(d => d.kind === 'videoinput'), selectedVideoDeviceId, 'Camera');
  populateDeviceSelect('micSelect', devices.filter(d => d.kind === 'audioinput'), selectedAudioDeviceId, 'Microphone');
  const speakerSel = document.getElementById('speakerSelect');
  if (speakerSel && !speakerRoutingSupported()) {
    speakerSel.innerHTML = '<option value="">System default (this browser cannot switch outputs)</option>';
    speakerSel.disabled = true;
  } else {
    populateDeviceSelect('speakerSelect', devices.filter(d => d.kind === 'audiooutput'), selectedSpeakerDeviceId, 'Speaker');
  }
}

function populateDeviceSelect(id, devices, selectedId, kind) {
  const sel = document.getElementById(id);
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = '';
  if (!devices.length) {
    const o = document.createElement('option');
    o.value = ''; o.textContent = `No ${kind.toLowerCase()} found`;
    sel.appendChild(o); sel.disabled = true;
    return;
  }
  sel.disabled = false;
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId;
    o.textContent = d.label || `${kind} ${i + 1}`;
    sel.appendChild(o);
  });
  const want = (selectedId && devices.some(d => d.deviceId === selectedId)) ? selectedId
             : (prev && devices.some(d => d.deviceId === prev)) ? prev : devices[0].deviceId;
  sel.value = want;
}

async function restartCapture() {
  try { if (mediaStream) mediaStream.getTracks().forEach(t => t.stop()); } catch (e) { /* ignore */ }
  mediaStream = null;
  cameraStartPromise = null;
  if (isLlmOnlyMode()) { await startAudioOnly(); }
  else { await startCamera(); }
}

function initDeviceControls() {
  const cam = document.getElementById('cameraSelect');
  const mic = document.getElementById('micSelect');
  if (cam) cam.addEventListener('change', async () => {
    selectedVideoDeviceId = cam.value;
    localStorage.setItem('studioCameraId', selectedVideoDeviceId);
    await restartCapture();
    startSettingsCameraPreview();   // re-attach the new stream to the preview
  });
  if (mic) mic.addEventListener('change', async () => {
    selectedAudioDeviceId = mic.value;
    localStorage.setItem('studioMicId', selectedAudioDeviceId);
    await restartCapture();
    if (_micTest) { stopMicTest(); toggleMicTest(); }   // re-test with the new mic
  });
  const speaker = document.getElementById('speakerSelect');
  if (speaker) speaker.addEventListener('change', () => {
    selectedSpeakerDeviceId = speaker.value;
    localStorage.setItem('studioSpeakerId', selectedSpeakerDeviceId);
    applySpeakerSink(currentAudioContext);   // reroute anything already speaking
  });
  const micTest = document.getElementById('micTestButton');
  if (micTest) micTest.addEventListener('click', toggleMicTest);
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', refreshDeviceLists);
  }
  refreshDeviceLists();
}

// ---- Settings: live camera preview + microphone test ----
function startSettingsCameraPreview() {
  const v = document.getElementById('settingsCameraPreview');
  const off = document.getElementById('settingsCameraOff');
  if (!v) return;
  const hasVideo = !!(mediaStream && mediaStream.getVideoTracks && mediaStream.getVideoTracks().length);
  if (hasVideo) {
    if (v.srcObject !== mediaStream) v.srcObject = mediaStream;
    v.style.display = '';
    if (off) off.style.display = 'none';
  } else {
    v.srcObject = null;
    v.style.display = 'none';
    if (off) {
      off.style.display = 'flex';
      off.textContent = isLlmOnlyMode() ? 'Camera is off — load a vision model to use it' : 'Starting camera…';
    }
    // For a vision-capable session, turn the camera on so the preview is live.
    if (!isLlmOnlyMode()) {
      try {
        startCamera().then(started => {
          if (started) startSettingsCameraPreview();
          else if (off) off.textContent = 'Camera unavailable — check browser permissions';
        }).catch(() => { if (off) off.textContent = 'Camera unavailable — check browser permissions'; });
      } catch (e) { /* ignore */ }
    }
  }
}
function stopSettingsCameraPreview() {
  const v = document.getElementById('settingsCameraPreview');
  if (v) v.srcObject = null;
}

let _micTest = null;
async function toggleMicTest() {
  if (_micTest) { stopMicTest(); return; }
  const btn = document.getElementById('micTestButton');
  const bar = document.getElementById('micLevelBar');
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: buildAudioConstraint() });
  } catch (e) {
    if (btn) { btn.textContent = 'Mic unavailable'; setTimeout(() => { if (btn && !_micTest) btn.textContent = 'Test microphone'; }, 2200); }
    return;
  }
  let ctx;
  try { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
  catch (e) { stream.getTracks().forEach(t => t.stop()); return; }
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  const state = { stream, ctx, raf: 0 };
  const tick = () => {
    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) { const dv = (data[i] - 128) / 128; sum += dv * dv; }
    const level = Math.min(100, Math.round(Math.sqrt(sum / data.length) * 240));
    if (bar) bar.style.width = level + '%';
    state.raf = requestAnimationFrame(tick);
  };
  state.raf = requestAnimationFrame(tick);
  _micTest = state;
  if (btn) { btn.textContent = 'Stop test'; btn.classList.add('testing'); }
}
function stopMicTest() {
  if (!_micTest) return;
  cancelAnimationFrame(_micTest.raf);
  try { _micTest.stream.getTracks().forEach(t => t.stop()); } catch (e) { /* ignore */ }
  try { _micTest.ctx.close(); } catch (e) { /* ignore */ }
  _micTest = null;
  const bar = document.getElementById('micLevelBar'); if (bar) bar.style.width = '0%';
  const btn = document.getElementById('micTestButton'); if (btn) { btn.textContent = 'Test microphone'; btn.classList.remove('testing'); }
}

// ---- Settings popup modal ----
function openSettings() {
  const modal = document.getElementById('settingsModal');
  if (!modal) return;
  // Close the mobile camera drawer if it's open.
  const ws = document.querySelector('.workspace');
  if (ws) ws.classList.remove('rail-open');
  const rt = document.getElementById('railToggle');
  if (rt) rt.textContent = '☰';
  modal.style.display = 'flex';
  refreshDeviceLists();
  const active = document.querySelector('.settings-tab.is-active');
  if (active && active.dataset.tab === 'devices') startSettingsCameraPreview();
}
function closeSettings() {
  const modal = document.getElementById('settingsModal');
  if (modal) modal.style.display = 'none';
  stopSettingsCameraPreview();
  stopMicTest();
}
function initSettingsModal() {
  const modal = document.getElementById('settingsModal');
  const open1 = document.getElementById('settingsButton');
  const open2 = document.getElementById('railSettingsButton');
  const close = document.getElementById('settingsModalClose');
  if (open1) open1.addEventListener('click', openSettings);
  if (open2) open2.addEventListener('click', openSettings);
  const homeInd = document.getElementById('homeModelIndicator');
  if (homeInd) homeInd.addEventListener('click', openSettings);
  if (close) close.addEventListener('click', closeSettings);
  if (modal) modal.addEventListener('click', (e) => { if (e.target === modal) closeSettings(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && modal.style.display === 'flex') closeSettings();
  });
}

// Settings are split into tabs — one section per tab. Clicking a tab shows its
// panel and hides the others. Hidden tabs (Hugging Face / RAG when unavailable)
// simply can't be selected.
function activateSettingsTab(key) {
  const tabs = document.querySelectorAll('.settings-tab');
  const panels = document.querySelectorAll('.settings-tab-panel');
  if (!tabs.length) return;
  tabs.forEach(t => t.classList.toggle('is-active', t.dataset.tab === key));
  panels.forEach(p => p.classList.toggle('is-active', p.dataset.tab === key));
  const content = document.querySelector('.settings-tab-content');
  if (content) content.scrollTop = 0;
  // The camera preview / mic test run only while the Devices tab is showing.
  if (key === 'devices') { startSettingsCameraPreview(); }
  else { stopSettingsCameraPreview(); stopMicTest(); }
  // Lazily list the Hugging Face models the first time the Add Model tab opens.
  if (key === 'huggingface') ensureHubLoaded();
  // Keep the active tab in view within the strip/sidebar (esp. the mobile strip).
  const active = document.querySelector('.settings-tab.is-active');
  if (active && active.scrollIntoView) {
    try { active.scrollIntoView({ inline: 'center', block: 'nearest' }); } catch (e) { /* ignore */ }
  }
}

function initSettingsTabs() {
  const tabs = document.querySelectorAll('.settings-tab');
  tabs.forEach(t => t.addEventListener('click', () => activateSettingsTab(t.dataset.tab)));
}

// Draggable chat / camera split — resize, collapse, reset, and persist.
function initRailResizer() {
  const resizer = document.getElementById('railResizer');
  const collapseBtn = document.getElementById('railCollapseBtn');
  if (!resizer) return;
  const MIN = 240, DEFAULT = 380;
  const maxWidth = () => Math.min(760, Math.round(window.innerWidth * 0.55));

  function applyWidth(px, save) {
    px = Math.max(MIN, Math.min(maxWidth(), px));
    document.documentElement.style.setProperty('--rail-w', px + 'px');
    if (save) localStorage.setItem('studioRailWidth', String(px));
    window.dispatchEvent(new Event('resize')); // re-fit the camera feed
  }
  function setCollapsed(on, save) {
    document.body.classList.toggle('rail-collapsed', on);
    if (collapseBtn) collapseBtn.textContent = on ? '⟨' : '⟩';
    if (save) localStorage.setItem('studioRailCollapsed', on ? '1' : '0');
    window.dispatchEvent(new Event('resize'));
  }

  // Restore saved layout.
  const savedW = parseInt(localStorage.getItem('studioRailWidth'), 10);
  if (Number.isFinite(savedW)) applyWidth(savedW, false);
  if (localStorage.getItem('studioRailCollapsed') === '1') setCollapsed(true, false);

  let dragging = false;
  const move = (e) => {
    if (!dragging) return;
    const x = e.touches ? e.touches[0].clientX : e.clientX;
    applyWidth(window.innerWidth - x, true); // rail is on the right
    e.preventDefault();
  };
  const up = () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('dragging');
    document.body.classList.remove('resizing');
  };
  const down = (e) => {
    if (e.target.closest('.rail-collapse-btn')) return; // let the button click through
    if (document.body.classList.contains('rail-collapsed')) setCollapsed(false, true);
    dragging = true;
    resizer.classList.add('dragging');
    document.body.classList.add('resizing');
    e.preventDefault();
  };
  resizer.addEventListener('mousedown', down);
  resizer.addEventListener('touchstart', down, { passive: false });
  window.addEventListener('mousemove', move);
  window.addEventListener('touchmove', move, { passive: false });
  window.addEventListener('mouseup', up);
  window.addEventListener('touchend', up);
  resizer.addEventListener('dblclick', () => { setCollapsed(false, true); applyWidth(DEFAULT, true); });
  if (collapseBtn) {
    collapseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      setCollapsed(!document.body.classList.contains('rail-collapsed'), true);
    });
  }
}

// Dynamic camera container sizing
function setupCameraContainer() {
  const container = document.querySelector('.camera-preview-container');
  const cameraSection = document.getElementById('cameraSection');

  if (!container || !cameraSection || !cameraPreview) return;

  // Size the dock to the video's own aspect ratio, as large as fits. Filling
  // the (tall, narrow) dock outright cropped a 16:9 feed to a vertical slice —
  // the user must always see the full camera frame.
  const fillDock = () => {
    if (cameraPreview.videoWidth === 0 || cameraPreview.videoHeight === 0) return;
    const rect = cameraSection.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const availW = Math.max(200, Math.round(rect.width - 20));
    const availH = Math.max(150, Math.round(rect.height - 20));
    const aspect = cameraPreview.videoWidth / cameraPreview.videoHeight;
    let w = availW, h = Math.round(availW / aspect);
    if (h > availH) { h = availH; w = Math.round(availH * aspect); }
    container.style.width = `${w}px`;
    container.style.height = `${h}px`;
  };
  cameraPreview.addEventListener('loadedmetadata', fillDock);
  window.addEventListener('resize', fillDock);
  if (cameraPreview.videoWidth > 0) fillDock();
}

// Resize camera container for uploaded image
function resizeContainerForImage(imageElement) {
  const container = document.querySelector('.camera-preview-container');
  const cameraSection = document.getElementById('cameraSection');

  if (!container || !cameraSection || !imageElement) return;

  // Wait for image to load and get natural dimensions
  if (imageElement.complete) {
    calculateImageContainerSize();
  } else {
    imageElement.onload = calculateImageContainerSize;
  }

  function calculateImageContainerSize() {
    const imageWidth = imageElement.naturalWidth;
    const imageHeight = imageElement.naturalHeight;

    if (imageWidth === 0 || imageHeight === 0) return;

    // Get camera section dimensions
    const sectionRect = cameraSection.getBoundingClientRect();
    const sectionWidth = sectionRect.width - 40; // Account for padding/margins
    const sectionHeight = sectionRect.height - 40;

    // Calculate aspect ratio
    const imageAspect = imageWidth / imageHeight;

    // Calculate optimal container size that fits within camera section
    let containerWidth = sectionWidth;
    let containerHeight = containerWidth / imageAspect;

    // If height is too tall, scale by height instead
    if (containerHeight > sectionHeight) {
      containerHeight = sectionHeight;
      containerWidth = containerHeight * imageAspect;
    }

    // Apply minimum sizes
    containerWidth = Math.max(containerWidth, 200);
    containerHeight = Math.max(containerHeight, 150);

    // Update container dimensions
    container.style.width = `${containerWidth}px`;
    container.style.height = `${containerHeight}px`;

    console.log(`Container resized for image: ${containerWidth}x${containerHeight} (aspect: ${imageAspect.toFixed(2)})`);
  }
}

// Clear uploaded image and return to webcam
function clearUploadedImage() {
  const imageOverlay = document.getElementById('imageOverlay');
  const imageCloseButton = document.getElementById('imageCloseButton');

  if (imageOverlay) {
    imageOverlay.remove();
  }
  if (imageCloseButton) {
    imageCloseButton.remove();
  }

  // Show camera preview again if not in logo mode
  if (cameraPreview && !isLlmOnlyMode()) {
    const imageCheckbox = document.getElementById('toggleImagePrompt');
    if (imageCheckbox && imageCheckbox.checked) {
      cameraPreview.style.display = 'block';

      // Restore camera container sizing by triggering setupCameraContainer
      if (cameraPreview.videoWidth > 0) {
        // Camera is active, trigger metadata event to recalculate size
        const event = new Event('loadedmetadata');
        cameraPreview.dispatchEvent(event);
      }
    }
  }

  console.log('Uploaded image cleared, returning to webcam view');
}


// Clear captured image and return to live camera feed
function clearCapturedImage() {
  // Clear by ID first (most recent elements)
  const capturedImageOverlay = document.getElementById('capturedImageOverlay');
  const capturedImageCloseButton = document.getElementById('capturedImageCloseButton');

  if (capturedImageOverlay) {
    capturedImageOverlay.remove();
  }
  if (capturedImageCloseButton) {
    capturedImageCloseButton.remove();
  }

  // Clear any remaining captured image elements by class/attributes to handle stacking
  const cameraPreviewContainer = document.querySelector('.camera-preview-container');
  if (cameraPreviewContainer) {
    // Remove any remaining image overlays with high z-index (captured images)
    const remainingOverlays = cameraPreviewContainer.querySelectorAll('img[style*="z-index: 5"]');
    remainingOverlays.forEach(overlay => overlay.remove());

    // Remove any remaining close buttons for captured images (but not uploaded images)
    // Only remove buttons that don't have the specific ID for uploaded images
    const remainingCloseButtons = cameraPreviewContainer.querySelectorAll('.image-close-button:not(#imageCloseButton)');
    remainingCloseButtons.forEach(button => button.remove());
  }
}

// Microphone functionality now handled by recordButton above

// New event handlers for dashboard buttons
sendButton.addEventListener('click', () => {
  sendTextMessage();
});

recordButton.addEventListener('click', () => {
  isMicrophoneMuted = !isMicrophoneMuted;
  toggleMicrophone(isMicrophoneMuted);
});

snapChatButton.addEventListener('click', () => {
  if (isLlmOnlyMode()) return; // Disabled in LLM-only mode
  captureAndAnimateSnap();
});

uploadButton.addEventListener('click', () => {
  if (isLlmOnlyMode()) return; // Disabled in LLM-only mode
  selectImage();
});

// Theme Toggle
themeToggle.addEventListener('click', () => {
  toggleTheme();
});

// Chat overlay buttons
const newChatButton = document.getElementById('newChatButton');
const abortButton = document.getElementById('abortButton');

newChatButton.addEventListener('click', () => {
  newChat();
});

const exportChatButton = document.getElementById('exportChatButton');
if (exportChatButton) {
  exportChatButton.addEventListener('click', () => exportChat());
}

abortButton.addEventListener('click', () => {
  abortResponse();
});

// Enter key support for message input
messageInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendTextMessage();
  }
});

if (systemPromptButton && systemPromptModal && systemPromptTextarea && systemPromptSave && systemPromptCancel) {
  systemPromptButton.addEventListener('click', openSystemPromptModal);
  systemPromptCancel.addEventListener('click', closeSystemPromptModal);
  systemPromptSave.addEventListener('click', saveSystemPrompt);
  systemPromptModal.addEventListener('click', handleSystemPromptBackdropClick);
  document.addEventListener('keydown', handleSystemPromptKeydown);
}

let mediaRecorder;
let audioBlob;

window.onload = function () {
  // Handle selected model vision capability.
  const isLlmOnly = isLlmOnlyMode();

  if (isLlmOnly) {
    // Disable vision-related elements
    const imagePromptCheckbox = document.getElementById('toggleImagePrompt');
    if (imagePromptCheckbox) {
      imagePromptCheckbox.checked = false;
      imagePromptCheckbox.disabled = true;
    }

    // Disable vision-related buttons in new UI
    if (snapChatButton) {
      snapChatButton.disabled = true;
      snapChatButton.title = 'Vision features disabled - LLM-only mode';
    }

    if (uploadButton) {
      uploadButton.disabled = true;
      uploadButton.title = 'Vision features disabled - LLM-only mode';
    }

  }

  // Camera/microphone capture starts after the model manager resolves the active
  // model. This avoids racing an audio-only request against a VLM camera request.

  // Initialize dashboard functionality
  initializeVoiceSync();
  initializeUtteranceSpeedControl();
  initStudioModelManager();
  initFontControls();
  initAccentControls();
  try { const cy = document.getElementById('copyYear'); if (cy) cy.textContent = new Date().getFullYear(); } catch (e) { /* ignore */ }
  initHubControls();
  initTtsToggle();
  initSpokenHighlightToggle();
  initFullscreenButton();
  initGenerationControls();
  initDeviceControls();
  initSettingsModal();
  initSettingsTabs();
  initModelCardModal();
  initModelManage();
  initRailResizer();
  initCameraCollapse();
  initAttachToggle();
  initVision();
  initBenchmark();
  initShowcase();
  initSolutions();
  initShutdownButton();
  initRagInspect();
  initVersionModal();
  hideRagControlsIfDisabled();
  if (isRagEnabled()) {
    initializeRagHealth();
  }
  fetchSystemPrompt();

  // Show settings rows that are hidden by default
  showSettingsRows();

  // Initialize camera display mode
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');
  if (includeImageCheckbox) {
    toggleCameraDisplay(includeImageCheckbox.checked);
  }

  // Initialize chat history checkbox behavior
  // Default: enabled for LLM-only, disabled for VLM (to avoid slow multi-image history)
  const chatHistoryCheckbox = document.getElementById('toggleChatHistory');
  if (chatHistoryCheckbox) {
    // LLM-only enables history by default; VLM disables it (multi-image history is slow).
    chatHistoryCheckbox.checked = isLlmOnly;
    chatHistoryCheckbox.addEventListener('change', handleChatHistoryToggle);
  }
  // No greeting message on load — the empty-state home (with suggestions) shows instead.
};

function toggleMicrophone(mute) {
  recordIcon.src = mute ? recordIconUrls.muted : recordIconUrls.active;

  if (audioTracks.length > 0) {
    audioTracks[0].enabled = !mute;
  }

  if (!mute) {
    recordButton.classList.add('recording');
    startRecording();
  } else {
    recordButton.classList.remove('recording');
    stopRecording();
  }
}

function updateSystemPromptButtonLabel() {
  if (!systemPromptButton) return;
  systemPromptButton.textContent = currentSystemPrompt ? 'Edit System Prompt' : 'Set System Prompt';
}

function openSystemPromptModal() {
  if (!systemPromptModal || !systemPromptTextarea) return;
  if (systemPromptRequestInFlight) return;

  systemPromptTextarea.value = currentSystemPrompt;
  if (systemPromptModalMessage) {
    systemPromptModalMessage.textContent = '';
    systemPromptModalMessage.classList.remove('error');
  }
  systemPromptModal.style.display = 'flex';
  setTimeout(() => systemPromptTextarea.focus(), 50);
}

function closeSystemPromptModal() {
  if (!systemPromptModal) return;
  systemPromptModal.style.display = 'none';
  if (systemPromptModalMessage) {
    systemPromptModalMessage.textContent = '';
    systemPromptModalMessage.classList.remove('error');
  }
}

function handleSystemPromptBackdropClick(event) {
  if (event.target === systemPromptModal) {
    closeSystemPromptModal();
  }
}

function handleSystemPromptKeydown(event) {
  if (event.key === 'Escape' && systemPromptModal && systemPromptModal.style.display === 'flex') {
    closeSystemPromptModal();
  }
}

async function fetchSystemPrompt() {
  if (!systemPromptButton) return;
  try {
    systemPromptRequestInFlight = true;
    systemPromptButton.disabled = true;
    const response = await fetch('/system-prompt');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    currentSystemPrompt = (data && typeof data.system_prompt === 'string') ? data.system_prompt : '';
    updateSystemPromptButtonLabel();
  } catch (error) {
    console.error('System prompt fetch failed:', error);
  } finally {
    systemPromptRequestInFlight = false;
    systemPromptButton.disabled = false;
  }
}

async function saveSystemPrompt() {
  if (!systemPromptTextarea || systemPromptRequestInFlight) return;
  const newPrompt = systemPromptTextarea.value.trim();

  try {
    systemPromptRequestInFlight = true;
    if (systemPromptButton) systemPromptButton.disabled = true;
    if (systemPromptSave) systemPromptSave.disabled = true;
    if (systemPromptCancel) systemPromptCancel.disabled = true;
    if (systemPromptModalMessage) {
      systemPromptModalMessage.textContent = 'Saving...';
      systemPromptModalMessage.classList.remove('error');
    }

    const response = await fetch('/system-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system_prompt: newPrompt })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    currentSystemPrompt = (data && typeof data.system_prompt === 'string') ? data.system_prompt : '';
    updateSystemPromptButtonLabel();
    closeSystemPromptModal();
  } catch (error) {
    if (systemPromptModalMessage) {
      systemPromptModalMessage.textContent = 'Failed to save system prompt.';
      systemPromptModalMessage.classList.add('error');
    }
    console.error('System prompt save failed:', error);
  } finally {
    systemPromptRequestInFlight = false;
    if (systemPromptButton) systemPromptButton.disabled = false;
    if (systemPromptSave) systemPromptSave.disabled = false;
    if (systemPromptCancel) systemPromptCancel.disabled = false;
  }
}
function getSupportedMimeType() {
  const possibleTypes = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/wav'
  ];
  return possibleTypes.find(type => MediaRecorder.isTypeSupported(type)) || '';
}

function startRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    console.warn('Recorder already active; ignoring duplicate start.');
    return;
  }

  const mimeType = getSupportedMimeType();

  if (!mimeType) {
    console.error('No supported MIME type found for MediaRecorder.');
    return;
  }

  try {
    recordedChunks = [];

    // Force audio-only tracks for safer recording
    const audioStream = new MediaStream(mediaStream.getAudioTracks());
    console.log('Audio Stream:', audioStream);

    mediaRecorder = new MediaRecorder(audioStream, { mimeType });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onerror = (event) => {
      console.error('MediaRecorder encountered an error:', event.error);
    };

    mediaRecorder.onstop = saveRecording;
    mediaRecorder.start();
    console.log('Recording started with MIME type:', mimeType);
  } catch (error) {
    console.error('Failed to start recording:', error.message);
    console.error('Error name:', error.name);
    console.error('Error stack:', error.stack);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    console.log('Recording stopped...');
  }
}

function saveRecording() {
  audioBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
  captureAndAnimateSnap();
}

async function stop(stopAudioFlag = false) {
  if (stopAudioFlag) {
    stopAudio();
  }

  try {
    await fetch('/stop', { method: 'POST' });
    console.log('Sent stop request to backend');
  } catch (err) {
    console.error('Failed to send stop request:', err);
  } finally {
    activeGeneration = false;
  }
}

function enqueueRequest(task) {
  requestQueue = requestQueue
    .catch(() => undefined)
    .then(task);
  return requestQueue;
}

// Chat message management
let chatHistory = [];
let lastCapturedImageDataUrl = null;

// Helper function to get current image data URL for chat preview
function getCurrentImageDataUrl() {
  // First priority: use the last captured image from snap/capture if available
  if (lastCapturedImageDataUrl) {
    return lastCapturedImageDataUrl;
  }

  // Second priority: check for uploaded image
  const imageOverlay = document.getElementById('imageOverlay');
  if (imageOverlay && imageOverlay.style.display === 'block' && imageOverlay.src) {
    return imageOverlay.src;
  }

  // Third priority: check for captured image overlay (fallback)
  const capturedImageOverlay = document.getElementById('capturedImageOverlay');
  if (capturedImageOverlay && capturedImageOverlay.src) {
    return capturedImageOverlay.src;
  }

  // Last resort: capture current webcam frame
  const cameraPreview = document.getElementById('cameraPreview');
  if (cameraPreview && cameraPreview.videoWidth > 0) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = cameraPreview.videoWidth;
    canvas.height = cameraPreview.videoHeight;
    context.drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
  }

  return null;
}

function addChatMessage(message, isUser = false, includeImagePreview = false) {
  // Add text message first
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
  if (isUser) {
    // User text is shown verbatim (escaped) to avoid rendering user-supplied markup.
    messageDiv.textContent = message;
  } else {
    // Assistant/system text may contain Markdown; render and sanitize it.
    const textSpan = document.createElement('span');
    textSpan.className = 'message-text';
    renderMarkdownInto(textSpan, message);
    messageDiv.appendChild(textSpan);
  }

  chatMessages.appendChild(messageDiv);

  // Add image preview as separate message after text if needed
  if (isUser && includeImagePreview) {
    const imageDataUrl = getCurrentImageDataUrl();
    if (imageDataUrl) {
      const imageMessageDiv = document.createElement('div');
      imageMessageDiv.className = 'message user image-message';

      const imagePreview = document.createElement('img');
      imagePreview.className = 'message-image-preview';
      imagePreview.src = imageDataUrl;
      imagePreview.alt = 'Image used in query';
      imagePreview.style.cssText = `
        display: block;
        width: 100%;
        height: auto;
        max-width: 150px;
        max-height: 150x;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        cursor: pointer;
        object-fit: cover;
      `;

      // Add click handler to open modal
      imagePreview.addEventListener('click', (e) => {
        e.stopPropagation();
        openImageModal(imageDataUrl);
      });

      imageMessageDiv.appendChild(imagePreview);
      chatMessages.appendChild(imageMessageDiv);
    }
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Store in history
  chatHistory.push({ message, isUser, timestamp: Date.now() });
}

// Helper function to create assistant message placeholder
function createAssistantMessage() {
  currentGenTokens = 0;   // fresh reply — reset the streamed-token counter
  const assistantMessage = document.createElement('div');
  assistantMessage.className = 'message assistant streaming-text';
  assistantMessage.textContent = 'Processing...';


  chatMessages.appendChild(assistantMessage);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Show abort button when assistant message is created
  showAbortButton();
}

// Helper function to clear messages if in single-shot mode or pending context clear
function clearIfSingleShotMode() {
  const chatHistoryCheckbox = document.getElementById('toggleChatHistory');
  const shouldClear = (chatHistoryCheckbox && !chatHistoryCheckbox.checked) || window.pendingContextClear;

  if (shouldClear) {
    // Clear previous messages (single-shot mode or context limit was hit)
    const messages = chatMessages.querySelectorAll('.message');
    messages.forEach(message => message.remove());
    chatHistory = [];

    // Reset the pending clear flag
    if (window.pendingContextClear) {
      window.pendingContextClear = false;
      console.log('UI cleared after context limit reset');
    }

    return true;
  }
  return false;
}

function sendTextMessage() {
  const message = messageInput.value.trim();
  if (!message) return;

  // No model loaded yet — prompt the user to pick/download one instead of failing.
  if (!getSelectedChatModel()) {
    addChatMessage('No model is loaded. Pick a model in **Settings → Model** and press **Load** (or download one from Hugging Face) to start chatting.', false);
    return;
  }

  // Clear input first
  messageInput.value = '';

  // Check if image should be included
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');

  if (includeImageCheckbox && includeImageCheckbox.checked && !isLlmOnlyMode()) {
    // Image mode: capture new image with the text message
    // Note: clearIfSingleShotMode is called inside captureAndAnimateSnap
    captureAndAnimateSnap(message);
  } else {
    // Text-only mode: clear if single-shot, then add message and process
    clearIfSingleShotMode();
    addChatMessage(message, true, false); // No image preview for text-only
    startProcessing('', message);
  }
}

// A suggested-prompt chip: fill the input and submit right away when a model is
// loaded; otherwise just place it in the (locked) input.
function useSuggestion(el) {
  const input = document.getElementById('messageInput');
  if (!input || !el) return;
  input.value = el.getAttribute('data-q') || '';
  if (modelReady()) {
    sendTextMessage();
  } else {
    input.focus();
  }
}

function clearChatHistory() {
  chatHistory = [];
  // Clear the stored captured image
  lastCapturedImageDataUrl = null;

  // Remove only chat messages, preserve overlay buttons
  const messages = chatMessages.querySelectorAll('.message');
  messages.forEach(message => message.remove());
}

// Handle chat history checkbox toggle
function handleChatHistoryToggle(event) {
  if (!event.target.checked) {
    // Checkbox was disabled - clear everything
    console.log('Chat history disabled - clearing history');

    // Clear UI messages
    clearChatHistory();

    // Clear backend history
    fetch('/clear-history', { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        console.log('Backend conversation history cleared:', data);
      })
      .catch(error => {
        console.error('Failed to clear backend history:', error);
      });
    // History disabled: the cleared chat shows the empty-state home again.
  } else {
    // Checkbox was re-enabled - keep user conversation
    console.log('Chat history enabled - conversations will accumulate');

    // If there's no actual conversation yet, return to the empty-state home.
    const userMessages = chatMessages.querySelectorAll('.message.user');
    if (userMessages.length === 0) {
      chatMessages.querySelectorAll('.message').forEach(message => message.remove());
    }
  }
}

// New Chat functionality
function newChat() {
  // Clear chat history
  clearChatHistory();

  // Clear backend conversation history
  fetch('/clear-history', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      console.log('Backend conversation history cleared:', data);
    })
    .catch(error => {
      console.error('Failed to clear backend history:', error);
    });

  // Don't reset settings - preserve user preferences
  // Only clear status messages
  clearStatusMessages();

  // Clear any uploaded images and captured images, return to webcam
  clearUploadedImage();
  clearCapturedImage();

  // Ensure settings rows remain visible
  showSettingsRows();

  // New chat returns to the empty-state home (hero + suggestions).

  // Hide abort button if visible
  hideAbortButton();
}

// Export the current conversation to a downloaded .log file — fully client-side.
// Reads the rendered messages straight from the DOM (streamed assistant replies
// build into the DOM, not chatHistory), so the export matches what's on screen.
function exportChat() {
  const container = document.getElementById('chatMessages');
  const nodes = container ? container.querySelectorAll('.message') : [];
  const body = [];
  let turns = 0;
  nodes.forEach((node) => {
    const who = node.classList.contains('user') ? 'You' : 'Assistant';
    if (node.classList.contains('image-message')) {
      body.push(who + ':', '[image]', '');
      turns++;
      return;
    }
    const textEl = node.querySelector('.message-text');
    const text = ((textEl ? textEl.textContent : node.textContent) || '').trim();
    if (!text) return;
    body.push(who + ':', text, '');
    turns++;
  });
  if (!turns) {
    alert('No chat to export yet — start a conversation first.');
    return;
  }
  const modelEl = document.getElementById('headerModelName');
  const model = modelEl && modelEl.textContent.trim() ? modelEl.textContent.trim() : '(none)';
  const now = new Date();
  const header = [
    'Neat GenAI Studio — chat export',
    'Exported: ' + now.toLocaleString(),
    'Model: ' + model,
    '='.repeat(60), '',
  ];
  const blob = new Blob([header.concat(body).join('\n')], { type: 'text/plain;charset=utf-8' });
  const pad = (n) => String(n).padStart(2, '0');
  const ts = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
             `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `neat-chat-${ts}.log`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1500);
}

// Reset all settings to defaults
function resetSettingsToDefaults() {
  // Reset image prompt checkbox
  const imagePromptCheckbox = document.getElementById('toggleImagePrompt');
  if (imagePromptCheckbox && !isLlmOnlyMode()) {
    imagePromptCheckbox.checked = true;
  }

  // Reset RAG checkbox
  const ragCheckbox = document.getElementById('toggleRAG');
  if (ragCheckbox) {
    ragCheckbox.checked = false;
  }

  // Reset transcription to automatic language detection.
  const languageSelect = document.getElementById('languageSelect');
  if (languageSelect) {
    languageSelect.value = 'auto';
    _detectedSpeechLanguage = 'en';
    updateVoiceEngineForLanguage();
  }

  // Clear any status messages
  const settingsMessage = document.getElementById('settingsMessage');
  if (settingsMessage) {
    settingsMessage.textContent = '';
  }

  const ragStatus = document.getElementById('ragStatus');
  if (ragStatus) {
    ragStatus.textContent = '';
  }

  // Ensure settings rows remain visible
  showSettingsRows();
}

// Abort current response
function abortResponse() {
  // Stop audible playback immediately, but serialize the backend /stop with
  // subsequent sends. Otherwise a fast second request can start before this
  // stop finishes, and the late stop then clears the new response's TTS queue.
  stopAudio();
  enqueueRequest(() => stop(false));

  // Remove speaking indicator from current assistant message
  const assistantMessages = chatMessages.querySelectorAll('.message.assistant');
  const currentAssistantMessage = assistantMessages[assistantMessages.length - 1];
  if (currentAssistantMessage) {
    currentAssistantMessage.classList.remove('speaking', 'streaming-text');

    // Hide the audio visualizer
    currentAssistantMessage.classList.remove('audio-playing');
    const canvas = currentAssistantMessage.querySelector('.audio-visualizer');
    if (canvas) {
      canvas.style.display = 'none';
    }

    // Note how many tokens were produced before the user stopped.
    setMessageTokens(currentAssistantMessage, currentGenTokens);
  }

  // Hide the abort button
  hideAbortButton();
}

// Show/hide abort button helpers
function showAbortButton() {
  if (abortButton) {
    abortButton.style.display = 'flex';
  }
}

function hideAbortButton() {
  if (abortButton) {
    abortButton.style.display = 'none';
  }
}

// Show settings rows that are hidden by default
function showSettingsRows() {
  // The transcription language is selectable by default.
  const languageRow = document.getElementById('languageRow');
  if (languageRow) languageRow.style.display = 'block';

  refreshVoiceOptions();
}

// Clear only status messages, preserve settings values
function clearStatusMessages() {
  // Only clear temporary user feedback messages, not system status
  const settingsMessage = document.getElementById('settingsMessage');
  if (settingsMessage) {
    settingsMessage.textContent = '';
  }

  // Keep ragStatus - it contains important system information about RAG database
  // that should persist across chat sessions
}


function getUtteranceSpeed() {
  const speedRange = document.getElementById('utteranceSpeedRange');
  const speed = speedRange ? parseFloat(speedRange.value) : 1.0;
  return Number.isFinite(speed) && speed > 0 ? speed : 1.0;
}

function updateUtteranceSpeedValue() {
  const speedValue = document.getElementById('utteranceSpeedValue');
  if (speedValue) {
    speedValue.textContent = `${getUtteranceSpeed().toFixed(2)}x`;
  }
}

function initializeUtteranceSpeedControl() {
  const speedRange = document.getElementById('utteranceSpeedRange');
  if (!speedRange) return;
  updateUtteranceSpeedValue();
  speedRange.addEventListener('input', updateUtteranceSpeedValue);
}
function captureAndAnimateSnap(textchat = null) {
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');

  // 1. Handle cases where no image is needed (early exit)
  if (isLlmOnlyMode() || !includeImageCheckbox.checked) {
    // Clear previous messages if in single-shot mode (BEFORE adding new message)
    clearIfSingleShotMode();
    if (textchat) {
      addChatMessage(textchat, true, false);
    }
    startProcessing('', textchat, !textchat);
    return;
  }

  // 2. Determine the final query text ONCE at the beginning
  let queryText = textchat;
  const isAudioQuery = !queryText && !!audioBlob;

  if (!queryText && !isAudioQuery) {
    queryText = 'Describe what you see in the picture.'; // Default for snap button case
  }

  // 3. Capture the image from the correct source
  clearCapturedImage();
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  const imageOverlay = document.getElementById('imageOverlay');

  const visionSize = getVisionImageSize();

  if (imageOverlay && imageOverlay.style.display === 'block') {
    const sourceWidth = imageOverlay.naturalWidth || visionSize?.width || 0;
    const sourceHeight = imageOverlay.naturalHeight || visionSize?.height || 0;

    if (visionSize && sourceWidth > 0 && sourceHeight > 0) {
      const targetWidth = visionSize.width;
      const targetHeight = visionSize.height;
      context.canvas.width = targetWidth;
      context.canvas.height = targetHeight;

      context.fillStyle = '#000';
      context.fillRect(0, 0, targetWidth, targetHeight);

      const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
      const drawWidth = Math.max(1, Math.round(sourceWidth * scale));
      const drawHeight = Math.max(1, Math.round(sourceHeight * scale));
      const offsetX = Math.round((targetWidth - drawWidth) / 2);
      const offsetY = Math.round((targetHeight - drawHeight) / 2);

      context.drawImage(
        imageOverlay,
        0,
        0,
        sourceWidth,
        sourceHeight,
        offsetX,
        offsetY,
        drawWidth,
        drawHeight
      );
    } else {
      context.canvas.width = Math.max(1, sourceWidth);
      context.canvas.height = Math.max(1, sourceHeight);
      context.drawImage(imageOverlay, 0, 0);
    }
  } else {
    const sourceWidth = cameraPreview.videoWidth;
    const sourceHeight = cameraPreview.videoHeight;

    if (visionSize && sourceWidth > 0 && sourceHeight > 0) {
      const targetWidth = visionSize.width;
      const targetHeight = visionSize.height;
      context.canvas.width = targetWidth;
      context.canvas.height = targetHeight;

      context.fillStyle = '#000';
      context.fillRect(0, 0, targetWidth, targetHeight);

      const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
      const drawWidth = Math.max(1, Math.round(sourceWidth * scale));
      const drawHeight = Math.max(1, Math.round(sourceHeight * scale));
      const offsetX = Math.round((targetWidth - drawWidth) / 2);
      const offsetY = Math.round((targetHeight - drawHeight) / 2);

      context.drawImage(
        cameraPreview,
        0,
        0,
        sourceWidth,
        sourceHeight,
        offsetX,
        offsetY,
        drawWidth,
        drawHeight
      );
    } else if (visionSize) {
      context.canvas.width = visionSize.width;
      context.canvas.height = visionSize.height;
      context.drawImage(
        cameraPreview,
        0,
        0,
        visionSize.width,
        visionSize.height,
        0,
        0,
        visionSize.width,
        visionSize.height
      );
    } else {
      context.canvas.width = sourceWidth;
      context.canvas.height = sourceHeight;
      context.drawImage(cameraPreview, 0, 0);
    }
  }

  lastCapturedImageDataUrl = canvas.toDataURL('image/png');

  snapAnimation.src = lastCapturedImageDataUrl;

  // Clear previous messages if in single-shot mode (BEFORE adding new message)
  clearIfSingleShotMode();

  // Add the user's message to the UI (unless it's an audio query)
  if (!isAudioQuery) {
    addChatMessage(queryText, true, true);
  }

  // Send the final request to the backend immediately after capture.
  startProcessing('', isAudioQuery ? null : queryText, isAudioQuery);
}


function startProcessing(resultMessage, textchat = null, waitForTranscription = false) {
  return enqueueRequest(async () => {
    if (activeGeneration) {
      await stop(true);
    }
    await startProcessingInternal(resultMessage, textchat, waitForTranscription);
  });
}

async function startProcessingInternal(resultMessage, textchat = null, waitForTranscription = false) {
  stopAudio();

  // Reset metrics
  document.getElementById('firstTokenTime').textContent = '...';
  document.getElementById('tpsValue').textContent = '...';
  document.getElementById('rtfValue').textContent = '...';
  document.getElementById('transcribeTime').textContent = '...';
  document.getElementById('asrLanguage').textContent = '—';
  document.getElementById('asrNoSpeech').textContent = '—';
  document.getElementById('asrLogprob').textContent = '—';
  document.getElementById('firstAudioTime').textContent = '...';

  // Reset First Audio timing
  userInputStartTime = Date.now();
  firstAudioStarted = false;

  // Keep audio muted until first token of the new generation arrives.
  // This drops late chunks from the previous generation during barge-in.
  shouldPlayAudio = false;
  pendingNewGenerationAudio = true;

  // Create placeholder for assistant response (unless waiting for transcription)
  if (!waitForTranscription) {
    createAssistantMessage();
  }

  const searchRag = document.getElementById('toggleRAG');
  const includeChatHistory = document.getElementById('toggleChatHistory');
  const formData = new FormData();

  const pendingAudio = audioBlob;
  audioBlob = null;
  if (!textchat && pendingAudio) {
    formData.append('audio_data', pendingAudio, 'audio.webm');
  }

  if (textchat) {
    formData.append('textchat', textchat);
  } else if (!pendingAudio) {
    formData.append('textchat', 'Describe what you see in the picture.');
  }

  formData.append('searchRag', isRagEnabled() && searchRag ? searchRag.checked : false);
  formData.append('includeChatHistory', includeChatHistory ? includeChatHistory.checked : true);
  formData.append('chatModel', getSelectedChatModel());
  formData.append('utteranceSpeed', getUtteranceSpeed().toFixed(2));
  formData.append('enableTts', isTtsEnabled());
  formData.append('maxTokens', getMaxTokens());
  formData.append('noThink', !getThinkingEnabled());   // disable reasoning when the toggle is off

  const sendRequest = async () => {
    activeGeneration = true;
    receivedEndSignal = false;
    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      updateAsrMetrics(data.asr);
      displayResult(data.question || resultMessage, 'static/sample_audio.wav', data.ttt);
      if (data.ignored) {
        activeGeneration = false;
        receivedEndSignal = true;
        pendingNewGenerationAudio = false;
        shouldPlayAudio = false;
        hideAbortButton();
        const message = data.message || 'No speech detected. Please try again.';
        if (typeof isVisionOpen === 'function' && isVisionOpen()) setVisionAskHint(message);
        addChatMessage(message, false, false);
        return;
      }
      const ragStatus = document.getElementById("ragStatus");
      if (isRagEnabled() && ragStatus) {
        ragStatus.textContent = data.rag_used
          ? `RAG used: yes, hits: ${data.rag_hits || 0}`
          : "RAG used: no";
      }
    } catch (error) {
      activeGeneration = false;
      console.error('Error uploading files:', error);
      displayResult('Error processing request', 'static/sample_audio.wav');
    }
  };

  const includeImageCheckbox = document.getElementById('toggleImagePrompt');
  const languageSelect = document.getElementById('languageSelect');
  const selectedLanguage = languageSelect ? languageSelect.value : 'en';

  formData.append('language', selectedLanguage);
  formData.append('responseLanguage', getSelectedVoiceLanguage());

  if (includeImageCheckbox && includeImageCheckbox.checked && snapAnimation.src) {
    try {
      const res = await fetch(snapAnimation.src);
      const originalBlob = await res.blob();
      formData.append('image_data', originalBlob, 'captured_image.png');
      await sendRequest();
    } catch (error) {
      activeGeneration = false;
      console.error('Error preparing image blob:', error);
      displayResult('Error preparing image', 'static/sample_audio.wav');
    }
  } else {
    await sendRequest();
  }
}

// Brutally clean up all global audio handles
function stopAudio() {
  audioEpoch++;                       // invalidate every in-flight / queued audio task
  shouldPlayAudio = false;
  // A late text 'update' must not flip audio back on for the aborted turn.
  pendingNewGenerationAudio = false;
  isPlaying = false;
  audioQueue.length = 0;
  nextStartTime = 0;
  // Cancel any in-flight / queued browser (Web Speech) utterances too.
  try { if (window.speechSynthesis) window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
  clearTtsHighlight();

  scheduledSources.forEach(source => {
    try { source.stop(0); source.disconnect(); }
    catch (e) { console.warn("Error stopping source node:", e); }
  });
  scheduledSources = [];
  scheduledHighlightTimers.forEach(timer => clearTimeout(timer));
  scheduledHighlightTimers = [];

  console.log("🧹 Audio playback completely stopped and cleaned up.");
}

/* ---- TTS "now speaking" highlight -------------------------------------
 * Highlights the sentence currently being uttered. Uses the CSS Custom
 * Highlight API so the rendered Markdown DOM is never mutated — the streaming
 * re-render replaces innerHTML, which would wipe any <mark> wrappers, but a
 * Highlight is just a set of Ranges we re-derive after each render. Degrades to
 * a no-op where the API is unavailable (older browsers). */
let _ttsHi = { container: null, text: '', searchFrom: 0, lastStart: -1 };
function _ttsSupported() {
  return typeof CSS !== 'undefined' && CSS.highlights
    && typeof Highlight !== 'undefined' && typeof Range !== 'undefined';
}

// The element holding the answer being spoken (Vision overlay if open, else the
// last assistant chat bubble's text).
function ttsAnswerContainer() {
  if (typeof isVisionOpen === 'function' && isVisionOpen()) {
    const va = document.getElementById('visionAnswer');
    if (va && va.textContent) return va;
  }
  const msgs = chatMessages ? chatMessages.querySelectorAll('.message.assistant') : [];
  const last = msgs[msgs.length - 1];
  return last ? (last.querySelector('.message-text') || last) : null;
}

// Highlighting the spoken sentence is a user setting (off by default). It is
// best-effort: matching a spoken sentence back to heavily Markdown-formatted
// text is fuzzy, so it may not highlight every sentence.
function ttsHighlightEnabled() {
  const el = document.getElementById('toggleSpokenHighlight');
  if (el) return el.checked;
  // On by default: enabled unless the user has explicitly turned it off.
  try { return localStorage.getItem('studioSpokenHighlight') !== '0'; } catch (e) { return true; }
}

function initSpokenHighlightToggle() {
  const el = document.getElementById('toggleSpokenHighlight');
  if (!el) return;
  // On by default: only an explicit '0' unchecks it.
  try { el.checked = localStorage.getItem('studioSpokenHighlight') !== '0'; } catch (e) { el.checked = true; }
  el.addEventListener('change', () => {
    try { localStorage.setItem('studioSpokenHighlight', el.checked ? '1' : '0'); } catch (e) { /* ignore */ }
    if (!el.checked) clearTtsHighlight();
  });
}

function setTtsHighlight(container, text) {
  if (!ttsHighlightEnabled()) { clearTtsHighlight(); return; }
  if (!container || !text) { clearTtsHighlight(); return; }
  // A new chunk: search forward from just past the previous sentence's start so
  // a repeated identical sentence matches its NEXT occurrence. A container swap
  // (e.g. Vision overlay opened) restarts the search from the top.
  if (_ttsHi.container !== container) { _ttsHi.searchFrom = 0; _ttsHi.lastStart = -1; }
  else { _ttsHi.searchFrom = _ttsHi.lastStart + 1; }
  _ttsHi.container = container;
  _ttsHi.text = text;
  applyTtsHighlight();
}

function clearTtsHighlight() {
  _ttsHi = { container: null, text: '', searchFrom: 0, lastStart: -1 };
  if (_ttsSupported()) { try { CSS.highlights.delete('tts'); } catch (e) { /* ignore */ } }
}

// (Re)derive the Range for the active sentence and register the highlight.
// Called on each new chunk AND after every Markdown re-render (text nodes are
// recreated, so prior Ranges are stale). Uses a fixed searchFrom so re-applying
// the SAME chunk re-finds the same sentence; only a new chunk advances it.
function applyTtsHighlight() {
  if (!ttsHighlightEnabled() || !_ttsSupported() || !_ttsHi.container || !_ttsHi.text) return;
  const range = _findTextRange(_ttsHi.container, _ttsHi.text, _ttsHi.searchFrom);
  if (!range) { try { CSS.highlights.delete('tts'); } catch (e) { /* ignore */ } return; }
  _ttsHi.lastStart = range._startNorm;   // remembered so the NEXT chunk searches past it
  try { CSS.highlights.set('tts', new Highlight(range)); } catch (e) { /* ignore */ }
}

/* ---- Browser (Web Speech API) TTS -------------------------------------
 * The "Browser" voice engine speaks entirely on the device: the server sends
 * only the (already sanitized) sentence text, and we utter it here with
 * speechSynthesis. Honors the global stop (audioEpoch) and the spoken-sentence
 * highlight, just like the server-audio path. */
const BROWSER_LANG_MAP = {
  en: 'en-US', ja: 'ja-JP', zh: 'zh-CN', ko: 'ko-KR', es: 'es-ES', fr: 'fr-FR',
  pt: 'pt-BR', de: 'de-DE', it: 'it-IT', no: 'nb-NO', vi: 'vi-VN', nl: 'nl-NL',
  ru: 'ru-RU', hi: 'hi-IN', ar: 'ar-SA', tr: 'tr-TR', pl: 'pl-PL', sv: 'sv-SE',
};
let _browserTtsWarned = false;

function browserTtsSupported() {
  return typeof window !== 'undefined' && 'speechSynthesis' in window
    && typeof SpeechSynthesisUtterance !== 'undefined';
}

// The browser voice to use: the user's pick if it matches the language, else the
// best default the device has for that language.
function browserVoiceForLang(langCode) {
  if (!browserTtsSupported()) return null;
  const voices = window.speechSynthesis.getVoices() || [];
  if (!voices.length) return null;
  const bcp = (BROWSER_LANG_MAP[langCode] || langCode || 'en-US').toLowerCase();
  const base = bcp.split('-')[0];
  const picked = localStorage.getItem('studioBrowserVoice');
  if (picked) {
    const byName = voices.find(v => v.name === picked);
    if (byName && (byName.lang || '').toLowerCase().startsWith(base)) return byName;
  }
  return voices.find(v => (v.lang || '').toLowerCase() === bcp)
      || voices.find(v => (v.lang || '').toLowerCase().startsWith(base))
      || null;
}

function speakBrowserTts(text, lang) {
  if (!text || !text.trim()) return;
  if (!browserTtsSupported()) {
    if (!_browserTtsWarned) {
      _browserTtsWarned = true;
      console.warn('This browser has no Web Speech (speechSynthesis) support.');
    }
    return;
  }
  if (!shouldPlayAudio) return;
  const myEpoch = audioEpoch;                 // honor the global stop mechanism
  const u = new SpeechSynthesisUtterance(text);
  u.lang = BROWSER_LANG_MAP[lang] || lang || 'en-US';
  const voice = browserVoiceForLang(lang);
  if (voice) u.voice = voice;
  const speed = (typeof getUtteranceSpeed === 'function') ? getUtteranceSpeed() : 1;
  u.rate = Math.min(2, Math.max(0.5, speed || 1));
  u.onstart = () => {
    if (myEpoch !== audioEpoch || !shouldPlayAudio) {
      try { window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
      return;
    }
    // Time to first audio: measured from send to when speech actually begins —
    // the same metric the server-audio path records at source.start().
    if (!firstAudioStarted && userInputStartTime) {
      const firstAudioTime = (Date.now() - userInputStartTime) / 1000;
      const el = document.getElementById('firstAudioTime');
      if (el) el.textContent = firstAudioTime.toFixed(2) + 's';
      firstAudioStarted = true;
    }
    const container = (typeof ttsAnswerContainer === 'function') ? ttsAnswerContainer() : null;
    if (container) setTtsHighlight(container, text);
  };
  u.onend = () => { if (myEpoch === audioEpoch) clearTtsHighlight(); };
  u.onerror = () => { if (myEpoch === audioEpoch) clearTtsHighlight(); };
  try { window.speechSynthesis.speak(u); } catch (e) { console.warn('speechSynthesis.speak failed', e); }
}

// Collapse runs of whitespace to a single space, keeping a map from each output
// index back to the index in the original string.
function _normalizeForMatch(s) {
  const out = []; const map = [];
  let prevSpace = false;
  for (let i = 0; i < s.length; i++) {
    if (/\s/.test(s[i])) {
      if (prevSpace) continue;
      out.push(' '); map.push(i); prevSpace = true;
    } else {
      out.push(s[i]); map.push(i); prevSpace = false;
    }
  }
  return { norm: out.join(''), map };
}

function _locateHay(segs, hayIdx) {
  for (const s of segs) {
    if (hayIdx >= s.start && hayIdx <= s.start + s.len) return { node: s.node, offset: hayIdx - s.start };
  }
  const last = segs[segs.length - 1];
  return last ? { node: last.node, offset: last.len } : null;
}

// Find `needle` among the container's text nodes (whitespace-insensitive,
// case-insensitive) and return a DOM Range spanning it, or null.
function _findTextRange(container, needle, fromNorm) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
  const segs = []; let hay = ''; let n;
  while ((n = walker.nextNode())) {
    const t = n.nodeValue || '';
    if (!t) continue;
    segs.push({ node: n, start: hay.length, len: t.length });
    hay += t;
  }
  if (!hay) return null;
  const H = _normalizeForMatch(hay);
  const needleNorm = _normalizeForMatch(needle).norm.trim();
  if (needleNorm.length < 2) return null;
  const hayL = H.norm.toLowerCase();
  const needL = needleNorm.toLowerCase();
  let pos = hayL.indexOf(needL, Math.min(Math.max(0, fromNorm | 0), hayL.length));
  if (pos < 0) pos = hayL.indexOf(needL);   // sentence may sit before the cursor
  if (pos < 0) return null;
  const startHay = H.map[pos];
  const endHay = H.map[pos + needL.length - 1] + 1;
  const startLoc = _locateHay(segs, startHay);
  const endLoc = _locateHay(segs, endHay);
  if (!startLoc || !endLoc) return null;
  const range = document.createRange();
  try {
    range.setStart(startLoc.node, startLoc.offset);
    range.setEnd(endLoc.node, endLoc.offset);
  } catch (e) { return null; }
  range._startNorm = pos;
  return range;
}

function displayResult(text, audioSrc, ttt = 0) {
  // Only update metrics - text display is now handled by WebSocket events
  // The assistant message should already exist from startProcessing()
  document.getElementById('transcribeTime').textContent = ttt + 's';

  // Note: text parameter (data.question) is ignored since it's just echoing user input
  // Actual LLM response comes through handleTextUpdate() via WebSocket
}


// Shared by the Upload button and drag-and-drop: read an image file and show
// it in the camera dock as the pending image for the next vision prompt.
function loadChatImageFile(file) {
  // Do nothing in LLM-only mode
  if (isLlmOnlyMode()) return;

  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    let imageOverlay = document.getElementById('imageOverlay');

    // Create the image overlay element if it doesn't exist
    if (!imageOverlay) {
      imageOverlay = document.createElement('img');
      imageOverlay.id = 'imageOverlay';
      imageOverlay.style.position = 'absolute';
      imageOverlay.style.top = '0';
      imageOverlay.style.left = '0';
      imageOverlay.style.width = '100%';
      imageOverlay.style.height = '100%';
      imageOverlay.style.objectFit = 'contain';
      imageOverlay.style.zIndex = '1';
      cameraPreview.parentElement.appendChild(imageOverlay);
    }

    // Hide the video feed and display the image
    cameraPreview.style.display = 'none';
    imageOverlay.src = e.target.result;
    imageOverlay.style.display = 'block';

    // Reveal the camera dock so the preview is actually visible — it's
    // collapsed/logo-mode by default in the chat-first layout, which would
    // otherwise hide the uploaded image. On mobile, open the rail drawer.
    const camSection = document.getElementById('cameraSection');
    if (camSection) {
      camSection.classList.remove('collapsed', 'logo-mode');
      const cb = document.getElementById('cameraCollapseBtn');
      if (cb) { cb.title = 'Collapse camera'; cb.setAttribute('aria-label', cb.title); }
    }
    const ws = document.querySelector('.workspace');
    const railToggle = document.getElementById('railToggle');
    const mobile = railToggle && getComputedStyle(railToggle).display !== 'none';
    if (ws && mobile) {
      ws.classList.add('rail-open');
      railToggle.textContent = '✕';
    }

    // Resize container to fit image dimensions
    resizeContainerForImage(imageOverlay);

    // Create close button if it doesn't exist
    let imageCloseButton = document.getElementById('imageCloseButton');
    if (!imageCloseButton) {
      imageCloseButton = document.createElement('button');
      imageCloseButton.id = 'imageCloseButton';
      imageCloseButton.innerHTML = '×';
      imageCloseButton.className = 'image-close-button';
      imageCloseButton.onclick = clearUploadedImage;
      cameraPreview.parentElement.appendChild(imageCloseButton);
    }
  };
  reader.onerror = () => {
    console.error('Failed to read the image file.');
  };
  reader.readAsDataURL(file);
}

// Small phones collapse Snap + Upload behind a "+" in the composer to keep
// the text field wide; the toggle is display:none everywhere else.
function initAttachToggle() {
  const btn = document.getElementById('attachToggleButton');
  const container = document.querySelector('#chatInput .input-container');
  if (!btn || !container) return;
  btn.addEventListener('click', () => {
    const open = container.classList.toggle('attach-open');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  // Collapse again once either action is chosen.
  ['snapChatButton', 'uploadButton'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', () => {
      container.classList.remove('attach-open');
      btn.setAttribute('aria-expanded', 'false');
    });
  });
}

function selectImage() {
  // Do nothing in LLM-only mode
  if (isLlmOnlyMode()) return;

  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.onchange = (event) => loadChatImageFile(event.target.files[0]);
  input.click();
}

// ---- Drag-and-drop image upload -------------------------------------------
// Dropping an image anywhere on the page attaches it: onto the full-screen
// Vision stage when that view is open, otherwise into the camera dock (the
// same path as the Upload button). A full-page overlay gives feedback while
// dragging; non-image drops and drops while uploads are disabled are ignored.

const dropOverlay = document.getElementById('dropOverlay');
let dragDepth = 0;   // dragenter/dragleave fire per child element — track depth

function dragHasFiles(event) {
  const types = event.dataTransfer && event.dataTransfer.types;
  return !!types && Array.from(types).includes('Files');
}

function isVisionModalOpen() {
  const modal = document.getElementById('visionModal');
  return !!modal && modal.style.display !== 'none';
}

// Returns null when a dropped image would be accepted, otherwise the reason
// shown in the overlay. Mirrors the Upload button's disabled state so drag &
// drop can't sneak an image past the model/vision gating.
function imageDropBlockedReason() {
  if (isVisionModalOpen()) return null;   // the Vision view accepts images freely
  if (isLlmOnlyMode()) return 'Vision features disabled - LLM-only mode';
  if (uploadButton && uploadButton.disabled) {
    return uploadButton.title || 'Image upload is currently disabled.';
  }
  return null;
}

function showDropOverlay() {
  if (!dropOverlay) return;
  const reason = imageDropBlockedReason();
  const title = document.getElementById('dropOverlayTitle');
  const sub = document.getElementById('dropOverlaySub');
  dropOverlay.classList.toggle('is-blocked', !!reason);
  if (title) {
    title.textContent = reason ? 'Can’t use an image right now'
      : (isVisionModalOpen() ? 'Drop image to view' : 'Drop image to attach');
  }
  if (sub) sub.textContent = reason || 'PNG, JPEG, WebP…';
  dropOverlay.style.display = 'flex';
}

function hideDropOverlay() {
  dragDepth = 0;
  if (dropOverlay) dropOverlay.style.display = 'none';
}

document.addEventListener('dragenter', (event) => {
  if (!dragHasFiles(event)) return;
  event.preventDefault();
  dragDepth++;
  showDropOverlay();
});

document.addEventListener('dragover', (event) => {
  if (!dragHasFiles(event)) return;
  // preventDefault marks the page as a valid drop target — without it the
  // browser navigates away to the dropped file.
  event.preventDefault();
  event.dataTransfer.dropEffect = imageDropBlockedReason() ? 'none' : 'copy';
});

document.addEventListener('dragleave', (event) => {
  if (!dragHasFiles(event)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) hideDropOverlay();
});

document.addEventListener('drop', (event) => {
  if (!dragHasFiles(event)) return;
  event.preventDefault();
  hideDropOverlay();
  if (imageDropBlockedReason()) return;
  const files = Array.from(event.dataTransfer.files || []);
  const image = files.find((f) => f.type && f.type.startsWith('image/'));
  if (!image) return;
  if (isVisionModalOpen()) loadVisionImageFile(image);
  else loadChatImageFile(image);
});

// ---- Devkit board camera ---------------------------------------------------
// The backend can grab a still from a camera plugged into the devkit board
// itself (/dev/video*), unlike the live preview which is the *browser's*
// webcam. Fetch one frame and feed it through the same paths as an uploaded
// image (camera dock in the chat, stage in the Vision view).

async function fetchBoardCameraFrame(device) {
  const url = device
    ? `/board-camera/snapshot?device=${encodeURIComponent(device)}`
    : '/board-camera/snapshot';
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try { msg = (await resp.json()).error || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  const blob = await resp.blob();
  return new File([blob], 'board-camera.jpg', { type: blob.type || 'image/jpeg' });
}

const boardCamButton = document.getElementById('cameraBoardBtn');
if (boardCamButton) {
  boardCamButton.addEventListener('click', async () => {
    if (isLlmOnlyMode()) return;
    boardCamButton.disabled = true;
    try {
      loadChatImageFile(await fetchBoardCameraFrame());
    } catch (err) {
      console.error('Board camera snapshot failed:', err);
      alert(`Board camera: ${err.message}`);
    } finally {
      boardCamButton.disabled = false;
    }
  });
}

socket.on('audio_chunk', (data) => {
  // Drop audio when the user turned off spoken responses (also guards any
  // late/in-flight chunk from a previous generation).
  if (!isTtsEnabled()) {
    return;
  }
  // Reject chunks if audio playback was aborted
  if (!shouldPlayAudio) {
    console.log('Ignoring audio chunk - playback aborted');
    return;
  }

  const text = data.text;
  const audioData = data.audio;

  console.log('Received text & audio :', text);

  if (data.tps != null) { const t = document.getElementById('tpsValue'); if (t) t.textContent = data.tps; }
  if (data.rtf != null) { const r = document.getElementById('rtfValue'); if (r) r.textContent = data.rtf; }

  // Browser TTS: the server sent only the (already sanitized) sentence text —
  // speak it locally with the Web Speech API instead of playing server audio.
  if (data.browser) {
    speakBrowserTts(text || '', data.lang);
    return;
  }

  // Keep the spoken text with its audio so we can highlight the sentence being
  // uttered when this chunk actually starts playing.
  // Don't override shouldPlayAudio - respect abort state
  audioQueue.push({ audio: audioData, text: text || '' });
  processAudioQueue();
});

// Handle system audio (e.g., voice switch confirmation) regardless of panel
socket.on('system_audio', (data) => {
  const audioData = data.audio;
  if (!audioData) return;
  // Play system audio immediately, bypassing panel3 checks
  processSystemAudioOnce(audioData);
});

// Handle standalone 'tps' event
socket.on('tps', (data) => {
  if (data !== undefined) {
    document.getElementById('tpsValue').textContent = data.toFixed(2);
  }
});

function handleTextUpdate(data) {
  if (data && data.results) {
    const cleanText = data.results.replace(/<\/s>|<pad>|<0x[0-9A-Fa-f]+>/g, '');
    if (!cleanText) return;

    currentGenTokens++;   // one content delta ≈ one generated token

    if (pendingNewGenerationAudio) {
      shouldPlayAudio = true;
      pendingNewGenerationAudio = false;
    }

    // Find the current assistant message (last message with 'assistant' class)
    const assistantMessages = chatMessages.querySelectorAll('.message.assistant');
    const currentAssistantMessage = assistantMessages[assistantMessages.length - 1];

    if (!currentAssistantMessage) {
      console.warn('No assistant message found to update');
      return;
    }

    // Check if this is the first token (message shows "Processing...")
    if (currentAssistantMessage.textContent === 'Processing...') {
      // Create separate text container to preserve canvas
      const textSpan = document.createElement('span');
      textSpan.className = 'message-text';

      // Preserve canvas when restructuring
      const canvas = currentAssistantMessage.querySelector('.audio-visualizer');
      currentAssistantMessage.innerHTML = ''; // Clear everything

      // Add text container and canvas back
      currentAssistantMessage.appendChild(textSpan);
      if (canvas) {
        currentAssistantMessage.appendChild(canvas);
        console.log('🔧 Restructured message with separate text container');
      }
      currentAssistantMessage.classList.add('speaking');
    }

    // Split the reply into reasoning (<think>…</think>) and the answer. Reasoning
    // streams into a collapsible block above the answer, and the two are counted
    // separately (thinking tokens vs output tokens).
    const msg = currentAssistantMessage;
    const answerEl = msg.querySelector('.message-text') || msg;
    msg._fullRaw = (msg._fullRaw || '') + cleanText;
    const parts = splitThinking(msg._fullRaw);

    // The template-injected case (a </think> with no opening <think>) can only be
    // recognized once </think> arrives — deltas before it were tentatively counted
    // as output. On that transition, reclassify them as thinking so the counts and
    // the block's token badge are right (safe: a plain answer never emits </think>).
    if (parts.present && !msg._wasPresent) {
      msg._wasPresent = true;
      if (msg._fullRaw.indexOf('<think>') === -1) {
        msg._thinkTokens = (msg._thinkTokens || 0) + (msg._outTokens || 0);
        msg._outTokens = 0;
      }
    }

    if (parts.present) {
      const block = ensureThinkBlock(msg);
      setMarkdownThrottled(block._thinkText, parts.thinking);
      if (!parts.closed) {
        msg._thinkTokens = (msg._thinkTokens || 0) + 1;
      } else {
        msg._outTokens = (msg._outTokens || 0) + 1;
        if (!msg._thinkCollapsed && parts.answer.trim()) {
          block.open = false;   // auto-collapse once the answer begins
          msg._thinkCollapsed = true;
        }
      }
      if (block._thinkTokensEl) {
        const n = msg._thinkTokens || 0;
        block._thinkTokensEl.textContent = n + ' token' + (n === 1 ? '' : 's');
      }
    } else {
      msg._outTokens = (msg._outTokens || 0) + 1;
    }
    setMarkdownThrottled(answerEl, parts.answer);

    // Live output-token counter (thinking is counted in the block above).
    setMessageTokens(msg, msg._outTokens || 0);

    // Scroll chat to bottom
    scrollChatToBottom();
  }
}

// Helper function to scroll chat to bottom
function scrollChatToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateAsrMetrics(metadata) {
  // The server names the model that produced this transcript. It is the only
  // signal the browser gets when the active ASR was changed from outside the
  // UI (an operator calling /control/asr), so trust it over the cached name.
  if (metadata && metadata.model && metadata.model !== _asrActive) {
    _asrActive = metadata.model;
    updateAsrModelIndicator();
    if (typeof updateManageButtons === 'function') updateManageButtons();
  }
  if (!metadata || typeof metadata !== 'object') return;
  const language = metadata.language || '—';
  const noSpeech = metadata.no_speech_prob == null ? NaN : Number(metadata.no_speech_prob);
  const avgLogprob = metadata.avg_logprob == null ? NaN : Number(metadata.avg_logprob);
  const languageEl = document.getElementById('asrLanguage');
  const noSpeechEl = document.getElementById('asrNoSpeech');
  const logprobEl = document.getElementById('asrLogprob');
  if (languageEl) languageEl.textContent = language;
  if (noSpeechEl) {
    noSpeechEl.textContent = Number.isFinite(noSpeech) ? `${(noSpeech * 100).toFixed(1)}%` : '—';
  }
  if (logprobEl) {
    logprobEl.textContent = Number.isFinite(avgLogprob) ? avgLogprob.toFixed(2) : '—';
  }
  if (metadata.language_detected && metadata.language) {
    _detectedSpeechLanguage = metadata.tts_language || metadata.language;
    updateVoiceEngineForLanguage();
  }
}

function displayTranscribedQuery(payload) {
  const metadata = (payload && typeof payload === 'object') ? payload : null;
  const text = metadata ? metadata.text : payload;
  if (!text) return;

  updateAsrMetrics(metadata);

  // Display transcribed text as a user message in the chat interface
  // Check if image should be included with transcribed audio
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');
  const shouldShowImagePreview = includeImageCheckbox && includeImageCheckbox.checked && !isLlmOnlyMode();
  addChatMessage(text, true, shouldShowImagePreview);

  // Surface the spoken question inside the full-screen Vision view too, so a
  // voice-first user sees what was heard alongside the streamed answer.
  if (typeof isVisionOpen === 'function' && isVisionOpen()) setVisionQuestion(text);

  // Create the assistant message placeholder that was delayed for transcription
  createAssistantMessage();

  // Scroll chat to bottom
  scrollChatToBottom();
}

function handleTtfsUpdate(data) {
  if (data) {
    document.getElementById('firstTokenTime').textContent = data + 's';
  }
}

function handleTranscriptionTimeUpdate(data) {
  if (data) {
    document.getElementById('transcribeTime').textContent = data + 's';
  }
}

socket.on('update', handleTextUpdate);
socket.on('generation_error', (data) => {
  const assistantMessages = chatMessages.querySelectorAll('.message.assistant');
  const currentAssistantMessage = assistantMessages[assistantMessages.length - 1];
  if (currentAssistantMessage) {
    currentAssistantMessage.classList.remove('speaking', 'streaming-text');
  }
  const message = (data && data.message) || 'Response generation failed. Please try again.';
  addChatMessage(`⚠️ ${message}`, false, false);
});
socket.on('end', (data) => {
  console.log('Received end event:', data);
  receivedEndSignal = true;
  activeGeneration = false;
  pendingNewGenerationAudio = false;

  // Remove speaking indicator from current assistant message
  const assistantMessages = chatMessages.querySelectorAll('.message.assistant');
  const currentAssistantMessage = assistantMessages[assistantMessages.length - 1];
  if (currentAssistantMessage) {
    currentAssistantMessage.classList.remove('speaking', 'streaming-text');

    // Final Markdown render + syntax highlighting + copy buttons on the reply.
    // Cancel any pending throttle render first, else it fires next frame and
    // re-renders without the highlighting/copy-button enhancement.
    const textContainer = currentAssistantMessage.querySelector('.message-text');
    if (textContainer) {
      cancelPendingRender(textContainer);
      renderMarkdownStreaming(textContainer, textContainer._raw || textContainer.textContent);
      enhanceCodeBlocks(textContainer);
      // Final render recreated the nodes — keep the TTS highlight aligned while
      // any queued audio finishes speaking.
      if (_ttsHi.container === textContainer) applyTtsHighlight();
    }
    // Finalize the reasoning block, if the model produced one.
    const tb = currentAssistantMessage._thinkBlock;
    if (tb && tb._thinkText) {
      cancelPendingRender(tb._thinkText);
      renderMarkdownStreaming(tb._thinkText, tb._thinkText._raw || tb._thinkText.textContent);
      enhanceCodeBlocks(tb._thinkText);
    }
    addResponseCopyButton(currentAssistantMessage);   // copy the whole reply

    // Show how many output tokens this reply generated (thinking counted separately).
    setMessageTokens(currentAssistantMessage,
      currentAssistantMessage._outTokens != null ? currentAssistantMessage._outTokens : currentGenTokens);

    // Don't remove audio-playing class here - let actual audio end handle it
    // The 'end' event is for text streaming, not audio playback
  }

  // Don't hide abort button here - keep it visible during audio playback
  // hideAbortButton(); // Moved to audio completion
});

// Append/update a small footer on an assistant message with the token count
// (and tok/s when a rate is available).
function setMessageTokens(messageEl, count) {
  if (!messageEl || !count) return;
  let meta = messageEl.querySelector('.message-meta');
  if (!meta) {
    meta = document.createElement('div');
    meta.className = 'message-meta';
    messageEl.appendChild(meta);
  }
  const tpsEl = document.getElementById('tpsValue');
  const tps = tpsEl ? parseFloat(tpsEl.textContent) : NaN;
  const rate = Number.isFinite(tps) && tps > 0 ? ` · ${tps.toFixed(1)} tok/s` : '';
  meta.textContent = `${count} token${count === 1 ? '' : 's'}${rate}`;
}
socket.on('ttfs', handleTtfsUpdate);
socket.on('transcription-time', handleTranscriptionTimeUpdate);
socket.on('transcription', displayTranscribedQuery);

// Handle context limit reached - backend history already cleared
socket.on('context_full', (data) => {
  console.log('Context limit reached:', data);

  // Add warning message to UI
  addChatMessage("⚠️ Context limit reached. History has been cleared - your next message will start a fresh conversation.", false);

  // Set flag to clear UI on next message send
  window.pendingContextClear = true;
});

async function processAudioQueue() {
  console.log(`Processing audio queue... Current queue length: ${audioQueue.length}`);
  const conditions = [];
  if (!shouldPlayAudio) conditions.push('shouldPlayAudio is false');
  // Audio can always play in the new dashboard design
  if (isPlaying) conditions.push('already playing');
  if (audioQueue.length === 0) conditions.push('audioQueue is empty');

  if (conditions.length > 0) {
    console.log(`Audio queue processing aborted due to: ${conditions.join(', ')}`);

    if (!isPlaying) {
      audioQueue.length = 0;
    }
    return;
  }

  isPlaying = true;
  // Tie this playback to the current epoch; a Stop bumps the epoch and every
  // check below (after each await, and before start) aborts this task.
  const myEpoch = audioEpoch;

  const item = audioQueue.shift();
  // Back-compat: older pushes were raw bytes; new ones are {audio, text}.
  const data = (item && item.audio !== undefined) ? item.audio : item;
  const chunkText = (item && item.text) || '';
  const blob = new Blob([data], { type: 'audio/wav' });
  const arrayBuffer = await blob.arrayBuffer();

  // Check again if audio should still play after async operations
  if (!shouldPlayAudio || myEpoch !== audioEpoch) {
    console.log('Audio playback cancelled during processing');
    isPlaying = false;
    return;
  }

  try {
    await ensureAudioContext();
    const audioBuffer = await currentAudioContext.decodeAudioData(arrayBuffer);

    // Final check before starting playback
    if (!shouldPlayAudio || myEpoch !== audioEpoch) {
      console.log('Audio playback cancelled before starting');
      isPlaying = false;
      return;
    }

    const source = currentAudioContext.createBufferSource();
    const analyser = currentAudioContext.createAnalyser();
    const gainNode = currentAudioContext.createGain();

    source.buffer = audioBuffer;
    source.connect(analyser);
    analyser.connect(gainNode);
    gainNode.connect(currentAudioContext.destination);

    scheduledSources.push(source);
    const startAt = Math.max(currentAudioContext.currentTime, nextStartTime);
    source.start(startAt);
    nextStartTime = startAt + audioBuffer.duration;

    // Highlight when this scheduled chunk actually starts, not while an earlier
    // sentence is still playing.
    const highlightDelay = Math.max(0, (startAt - currentAudioContext.currentTime) * 1000);
    const highlightTimer = setTimeout(() => {
      scheduledHighlightTimers = scheduledHighlightTimers.filter(t => t !== highlightTimer);
      if (shouldPlayAudio && myEpoch === audioEpoch) {
        setTtsHighlight(ttsAnswerContainer(), chunkText);
      }
    }, highlightDelay);
    scheduledHighlightTimers.push(highlightTimer);

    // Track First Audio timing (only for the very first audio chunk)
    if (!firstAudioStarted && userInputStartTime) {
      const firstAudioTime = (Date.now() - userInputStartTime) / 1000;
      document.getElementById('firstAudioTime').textContent = firstAudioTime.toFixed(2) + 's';
      firstAudioStarted = true;
    }

    source.onended = () => {
      scheduledSources = scheduledSources.filter(s => s !== source);

      // Disconnect and clean up
      try {
        source.disconnect();
        analyser.disconnect();
        gainNode.disconnect();
      } catch (e) {
        console.warn('Error disconnecting nodes:', e);
      }
      console.log('Audio playback ended.');


      // Check if we should hide the abort button
      // Hide it only if text streaming has ended AND no more audio in queue
      if (receivedEndSignal && audioQueue.length === 0 && scheduledSources.length === 0) {
        hideAbortButton();
        clearTtsHighlight();   // nothing left to speak — drop the highlight
      }
    };
  } catch (err) {
    console.warn('AudioContext playback failed:', err);
    isPlaying = false;

    // Hide abort button if text streaming ended and this was the last audio
    if (receivedEndSignal && audioQueue.length === 0 && scheduledSources.length === 0) {
      hideAbortButton();
    }
  }
  isPlaying = false;
  processAudioQueue();
}

async function ensureAudioContext() {
  if (!currentAudioContext || currentAudioContext.state === 'closed') {
    currentAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    await applySpeakerSink(currentAudioContext);
  }
  if (currentAudioContext.state === 'suspended') {
    await currentAudioContext.resume();
  }
}

async function processSystemAudioOnce(data) {
  try {
    const blob = new Blob([data], { type: 'audio/wav' });
    const arrayBuffer = await blob.arrayBuffer();

    await ensureAudioContext();
    const audioBuffer = await currentAudioContext.decodeAudioData(arrayBuffer);

    const source = currentAudioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(currentAudioContext.destination);
    scheduledSources.push(source);
    const startAt = Math.max(currentAudioContext.currentTime, nextStartTime);
    source.start(startAt);
    nextStartTime = startAt + audioBuffer.duration;
    source.onended = () => {
      scheduledSources = scheduledSources.filter(s => s !== source);
      try { source.disconnect(); } catch (e) { /* already disconnected by stopAudio */ }
    };
  } catch (e) {
    console.warn('System audio playback failed:', e);
  }
}

function getSelectedVoiceLanguage() {
  const languageSelect = document.getElementById('languageSelect');
  const selected = languageSelect ? languageSelect.value : 'en';
  return selected === 'auto' ? (_detectedSpeechLanguage || 'en') : selected;
}

function getVoiceStorageKey(lang) {
  return `${lang}VoiceId`;
}

function setVoiceRowVisible(visible) {
  const voiceRow = document.getElementById('voiceRow');
  if (voiceRow) {
    // Keep the allowlisted server voice catalog reachable while Browser is
    // active so a language omitted during setup can be installed from the UI.
    const show = visible && _currentTtsEngine !== 'piper-plus';
    voiceRow.style.display = show ? 'block' : 'none';
  }
}

// Show only the voice-selection setting relevant to the active engine:
// piper-plus -> its voice picker; piper-tts -> the rhasspy voice picker;
// browser -> the device-voice picker.
function applyEngineVoiceVisibility() {
  const ppRow = document.getElementById('piperPlusVoiceRow');
  if (ppRow) {
    ppRow.style.display = (_currentTtsEngine === 'piper-plus' && _ppVoicesAvailable) ? '' : 'none';
  }
  const brRow = document.getElementById('browserVoiceRow');
  if (brRow) {
    const show = _currentTtsEngine === 'browser' && browserTtsSupported();
    brRow.style.display = show ? '' : 'none';
    if (show) populateBrowserVoices();
  }
  refreshVoiceOptions();   // recompute the rhasspy voice row (respects the engine)
}

// Fill the Browser-voice dropdown from the device's Web Speech voices for the
// selected language. speechSynthesis voices can load asynchronously, so this is
// also re-run on the 'voiceschanged' event.
function populateBrowserVoices() {
  const sel = document.getElementById('browserVoiceSelect');
  const status = document.getElementById('browserVoiceStatus');
  if (!sel) return;
  if (!browserTtsSupported()) {
    sel.innerHTML = '';
    if (status) status.textContent = 'This browser has no built-in speech voices.';
    return;
  }
  const lang = (typeof getSelectedVoiceLanguage === 'function') ? getSelectedVoiceLanguage() : 'en';
  const bcp = (BROWSER_LANG_MAP[lang] || lang || 'en-US').toLowerCase();
  const base = bcp.split('-')[0];
  const all = window.speechSynthesis.getVoices() || [];
  const matches = all.filter(v => (v.lang || '').toLowerCase().startsWith(base));
  const voices = matches.length ? matches : all;
  sel.innerHTML = '';
  if (!voices.length) {
    if (status) status.textContent = 'No device voices found yet — the browser may still be loading them.';
    return;
  }
  const stored = localStorage.getItem('studioBrowserVoice');
  voices.forEach(v => {
    const o = document.createElement('option');
    o.value = v.name;
    o.textContent = `${v.name} (${v.lang})${v.default ? ' · default' : ''}`;
    sel.appendChild(o);
  });
  sel.value = voices.some(v => v.name === stored) ? stored : voices[0].name;
  if (status) status.textContent = matches.length ? '' : 'No voice for this language — using a default device voice.';
  sel.onchange = () => {
    try { localStorage.setItem('studioBrowserVoice', sel.value); } catch (e) { /* ignore */ }
    if (status) status.textContent = 'Active';
  };
}

function setVoiceLabel(lang) {
  const voiceLabel = document.querySelector('label[for="voiceSelect"]');
  const languageSelect = document.getElementById('languageSelect');
  const selectedOption = languageSelect ? languageSelect.options[languageSelect.selectedIndex] : null;
  const label = languageSelect && languageSelect.value === 'auto'
    ? `${String(lang).toUpperCase()} (detected)`
    : (selectedOption ? selectedOption.textContent.replace(/\s*\([^)]*\)\s*$/, '') : lang);
  if (voiceLabel) {
    voiceLabel.textContent = `${label} Voice:`;
  }
}

async function refreshVoiceOptions() {
  const voiceSelect = document.getElementById('voiceSelect');
  const activateButton = document.getElementById('voiceActivateButton');
  if (!voiceSelect) return;

  const lang = getSelectedVoiceLanguage();
  setVoiceLabel(lang);

  try {
    const response = await fetch(`/voices?lang=${encodeURIComponent(lang)}`);
    if (!response.ok) throw new Error('Failed to load voices');

    const data = await response.json();
    const voices = Array.isArray(data.voices) ? data.voices : [];
    while (voiceSelect.firstChild) voiceSelect.removeChild(voiceSelect.firstChild);

    if (voices.length === 0) {
      voiceSelect.dataset.lang = lang;
      voiceSelect.dataset.prev = '';
      setVoiceRowVisible(false);
      if (activateButton) activateButton.style.display = 'none';
      return;
    }

    const stored = localStorage.getItem(getVoiceStorageKey(lang));
    const current = data.current;
    const toSelect = current || stored || voices[0].id;

    voices.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.id;
      opt.dataset.installed = v.installed ? 'true' : 'false';
      opt.textContent = `${v.label || v.id}${v.installed ? '' : ' · download'}`;
      voiceSelect.appendChild(opt);
    });

    const found = Array.from(voiceSelect.options).some(o => o.value === toSelect);
    voiceSelect.value = found ? toSelect : voiceSelect.options[0].value;
    voiceSelect.dataset.lang = lang;
    voiceSelect.dataset.prev = voiceSelect.value;
    setVoiceRowVisible(true);
    if (activateButton) {
      activateButton.style.display = '';
      const selected = voiceSelect.options[voiceSelect.selectedIndex];
      activateButton.textContent = selected?.dataset.installed === 'true'
        ? 'Use server voice'
        : 'Download and use server voice';
    }
  } catch (e) {
    voiceSelect.dataset.lang = lang;
    voiceSelect.dataset.prev = '';
    setVoiceRowVisible(false);
    if (activateButton) activateButton.style.display = 'none';
  }
}

// Settings are now always visible, so initialize voice sync on load
function initializeVoiceSync() {
  refreshVoiceOptions();
}

function setRagControls(dbReady) {
  const importButton = document.getElementById("importRagDatabaseButton");
  const uploadButton = document.getElementById("uploadToRagButton");
  if (importButton) importButton.disabled = !dbReady;
  if (uploadButton) uploadButton.disabled = false;
}

// ---- RAG database inspector -------------------------------------------------
// Shows the build summary + every ingested chunk (source, headers, text) so users
// can see exactly what's in the RAG DB they uploaded. All rendering uses
// textContent, so chunk text can never inject markup.
let _ragInspectDocs = [];

// About / version modal — shows the git commit, branch and date (from
// window.SIMA_CONFIG.version, injected by /config.js).
function initVersionModal() {
  const btn = document.getElementById('aboutVersionBtn');
  const modal = document.getElementById('versionModal');
  if (!btn || !modal) return;
  const close = document.getElementById('versionModalClose');
  const grid = document.getElementById('versionGrid');
  const hide = () => { modal.style.display = 'none'; };
  const open = () => {
    if (grid) {
      grid.textContent = '';
      const v = (window.SIMA_CONFIG && window.SIMA_CONFIG.version) || {};
      const rows = [
        ['Studio', v.name || 'Neat GenAI Studio'],
        ['Commit', (v.commit || 'unknown') + (v.dirty ? ' · modified' : '')],
        ['Branch', v.branch || 'unknown'],
      ];
      if (v.date) rows.push(['Date', v.date]);
      rows.forEach(([k, val]) => {
        const kk = document.createElement('div'); kk.className = 'version-k'; kk.textContent = k;
        const vv = document.createElement('div'); vv.className = 'version-v'; vv.textContent = val;
        grid.appendChild(kk); grid.appendChild(vv);
      });
    }
    modal.style.display = 'flex';
  };
  btn.addEventListener('click', open);
  if (close) close.addEventListener('click', hide);
  modal.addEventListener('click', (e) => { if (e.target === modal) hide(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') hide();
  });
}

function initRagInspect() {
  const btn = document.getElementById('inspectRagButton');
  const modal = document.getElementById('ragInspectModal');
  if (!btn || !modal) return;
  const close = document.getElementById('ragInspectClose');
  const filter = document.getElementById('ragInspectFilter');
  btn.addEventListener('click', openRagInspect);
  if (close) close.addEventListener('click', closeRagInspect);
  if (filter) filter.addEventListener('input', renderRagChunks);
  modal.addEventListener('click', (e) => { if (e.target === modal) closeRagInspect(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeRagInspect();
  });
}

function closeRagInspect() {
  const modal = document.getElementById('ragInspectModal');
  if (modal) modal.style.display = 'none';
  document.body.classList.remove('rag-inspect-open');
}

function openRagInspect() {
  const modal = document.getElementById('ragInspectModal');
  const summary = document.getElementById('ragInspectSummary');
  const body = document.getElementById('ragInspectBody');
  if (!modal) return;
  modal.style.display = 'flex';
  document.body.classList.add('rag-inspect-open');
  _ragInspectDocs = [];
  if (summary) summary.textContent = 'Loading…';
  if (body) body.textContent = '';
  const count = document.getElementById('ragInspectCount');
  if (count) count.textContent = '';
  fetch('/rag/inspect')
    .then((r) => r.json())
    .then((data) => {
      renderRagSummary(data);
      _ragInspectDocs = Array.isArray(data.documents) ? data.documents : [];
      renderRagChunks();
    })
    .catch((err) => {
      if (summary) summary.textContent = 'Could not load the RAG database: ' + err;
    });
}

function renderRagSummary(data) {
  const summary = document.getElementById('ragInspectSummary');
  if (!summary) return;
  summary.textContent = '';
  if (data && data.enabled === false) {
    summary.textContent = 'RAG is disabled.';
    return;
  }
  const meta = (data && data.meta) || {};
  const rows = [];
  if (meta.input) rows.push(['Source', String(meta.input)]);
  if (meta.embedding_model) {
    rows.push(['Embedding', String(meta.embedding_model).replace(/\/+$/, '').split('/').pop()]);
  }
  rows.push(['Chunks', String(data && data.count != null ? data.count : (meta.chunks || 0))]);
  if (data && data.path) rows.push(['File', String(data.path)]);
  rows.forEach(([k, v]) => {
    const row = document.createElement('div');
    row.className = 'rag-sum-row';
    const kk = document.createElement('span'); kk.className = 'rag-sum-k'; kk.textContent = k;
    const vv = document.createElement('span'); vv.className = 'rag-sum-v'; vv.textContent = v;
    row.appendChild(kk); row.appendChild(vv);
    summary.appendChild(row);
  });
  if (data && data.error) {
    const e = document.createElement('div');
    e.className = 'rag-sum-error';
    e.textContent = data.error;
    summary.appendChild(e);
  }
}

function renderRagChunks() {
  const body = document.getElementById('ragInspectBody');
  const countEl = document.getElementById('ragInspectCount');
  const filterEl = document.getElementById('ragInspectFilter');
  if (!body) return;
  body.textContent = '';
  const filter = (filterEl ? filterEl.value : '').trim().toLowerCase();
  const docs = _ragInspectDocs.filter((d) => {
    if (!filter) return true;
    const md = (d && d.metadata) || {};
    const hay = (String((d && d.text) || '') + ' ' + Object.values(md).join(' ')).toLowerCase();
    return hay.indexOf(filter) !== -1;
  });
  if (countEl) {
    countEl.textContent = docs.length + (filter ? ' / ' + _ragInspectDocs.length : '') + ' chunks';
  }
  if (!docs.length) {
    const empty = document.createElement('div');
    empty.className = 'rag-empty';
    empty.textContent = _ragInspectDocs.length ? 'No chunks match the filter.' : 'No chunks in the RAG database.';
    body.appendChild(empty);
    return;
  }
  docs.forEach((d, i) => {
    const md = (d && d.metadata) || {};
    const headers = Object.keys(md)
      .filter((k) => k.toLowerCase().indexOf('header') === 0 && md[k])
      .sort()
      .map((k) => md[k]);
    const card = document.createElement('div');
    card.className = 'rag-chunk';
    const head = document.createElement('div');
    head.className = 'rag-chunk-head';
    const idx = document.createElement('span');
    idx.className = 'rag-chunk-idx';
    idx.textContent = i + 1;
    const crumb = document.createElement('span');
    crumb.className = 'rag-chunk-crumb';
    crumb.textContent = headers.length ? headers.join(' › ') : '(no header)';
    head.appendChild(idx); head.appendChild(crumb);
    const text = document.createElement('div');
    text.className = 'rag-chunk-text';
    text.textContent = String((d && d.text) || '');   // plain text → safe from injection
    card.appendChild(head); card.appendChild(text);
    body.appendChild(card);
  });
}

// Initialize RAG health check
function initializeRagHealth(attempt = 1) {
  if (!isRagEnabled()) {
    return;
  }

  fetch("/raghealth")
    .then(res => res.json().then(data => ({ status: res.status, data })))
    .then(({ status, data }) => {
      const dbStatus = data.rag_db === "ok";

      // Update status message
      if (dbStatus) {
        ragServerStatusText = "✅ RAG Database is online.";
      } else {
        ragServerStatusText = "❌ RAG Database is not ready yet, please wait...";
      }

      console.log(ragServerStatusText);
      const ragStatus = document.getElementById("ragStatus");
      if (ragStatus) ragStatus.textContent = ragServerStatusText;

      setRagControls(dbStatus);
      if (!dbStatus && attempt < 15) {
        setTimeout(() => initializeRagHealth(attempt + 1), 2000);
      }
    })
    .catch(err => {
      ragServerStatusText = "❌ Error checking RAG server health.";
      console.error(ragServerStatusText, err);
      const ragStatus = document.getElementById("ragStatus");
      if (ragStatus) ragStatus.textContent = ragServerStatusText;

      setRagControls(false);
      if (attempt < 15) {
        setTimeout(() => initializeRagHealth(attempt + 1), 2000);
      }
    });
}


// Settings panel is always visible, no close button needed

// The transcription language selector is available by default.
// ---- Piper Plus voice picker (allowlisted multilingual models) -------------
async function initPiperPlusVoices() {
  const sel = document.getElementById('piperPlusVoiceSelect');
  const row = document.getElementById('piperPlusVoiceRow');
  if (!sel || !row) return;
  let data;
  try {
    data = await (await fetch('/piperplus/voices')).json();
  } catch (e) { _ppVoicesAvailable = false; row.style.display = 'none'; return; }
  const voices = (data && data.voices) || [];
  _ppVoicesAvailable = voices.length > 0;
  if (!voices.length) { row.style.display = 'none'; return; }
  sel.innerHTML = '';
  voices.forEach(v => {
    const o = document.createElement('option');
    o.value = v.key;
    o.textContent = `${v.label || v.key}${v.installed ? '' : ' · download'}`;
    sel.appendChild(o);
  });
  if (data.current) sel.value = data.current;
  applyEngineVoiceVisibility();   // show only if piper-plus is the active engine
  sel.onchange = async () => {
    const status = document.getElementById('piperPlusVoiceStatus');
    if (status) status.textContent = 'Loading…';
    sel.disabled = true;
    try {
      const r = await fetch('/piperplus/select', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: sel.value })
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || d.status !== 'ok') throw new Error((d && d.error) || 'failed to load');
      if (status) status.textContent = 'Active';
      initPiperPlusVoices();
    } catch (err) {
      if (status) status.textContent = 'Failed: ' + err.message;
    } finally {
      sel.disabled = false;
    }
  };
}

// ---- Voice engine picker (piper-plus vs rhasspy piper-tts) ----------------
// Shows ONLY the engine(s) that can speak the currently selected language, so a
// language is never offered an incompatible engine. When both engines support
// it, the user can choose which is preferred; when only one does, it's shown
// read-only. Re-runs whenever the language changes.
async function updateVoiceEngineForLanguage() {
  const sel = document.getElementById('voiceEngineSelect');
  const row = document.getElementById('voiceEngineRow');
  if (!sel || !row) return;
  const lang = getSelectedVoiceLanguage();
  let data;
  try {
    data = await (await fetch('/tts/engine?lang=' + encodeURIComponent(lang))).json();
  } catch (e) { row.style.display = 'none'; return; }
  const engines = (data && data.engines) || [];
  if (data.current) _currentTtsEngine = data.current;
  if (!engines.length) { row.style.display = 'none'; applyEngineVoiceVisibility(); return; }
  sel.innerHTML = '';
  engines.forEach(e => {
    const o = document.createElement('option');
    o.value = e.key;
    o.textContent = e.label || e.key;
    sel.appendChild(o);
  });
  sel.value = engines.some(e => e.key === _currentTtsEngine) ? _currentTtsEngine : engines[0].value;
  _currentTtsEngine = sel.value;
  sel.disabled = engines.length < 2;   // only one engine supports this language → read-only
  row.style.display = '';              // always show the supported engine(s)
  sel.onchange = async () => {
    const status = document.getElementById('voiceEngineStatus');
    if (status) status.textContent = 'Saving…';
    sel.disabled = true;
    try {
      const r = await fetch('/tts/engine', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine: sel.value })
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || d.status !== 'ok') throw new Error((d && d.error) || 'failed');
      _currentTtsEngine = sel.value;
      applyEngineVoiceVisibility();     // swap which voice picker is shown
      if (status) status.textContent = 'Active';
    } catch (err) {
      if (status) status.textContent = 'Failed: ' + err.message;
    } finally {
      sel.disabled = engines.length < 2;
    }
  };
  applyEngineVoiceVisibility();
}

window.addEventListener('DOMContentLoaded', () => {
  const langRow = document.getElementById('languageRow');
  if (langRow) langRow.style.display = 'block';
  initPiperPlusVoices();
  updateVoiceEngineForLanguage();   // also drives refreshVoiceOptions via applyEngineVoiceVisibility
  // Device speech voices often load asynchronously — repopulate the Browser
  // voice picker when they arrive (only relevant while that engine is active).
  if (browserTtsSupported() && window.speechSynthesis.addEventListener) {
    window.speechSynthesis.addEventListener('voiceschanged', () => {
      if (_currentTtsEngine === 'browser') populateBrowserVoices();
    });
    try { window.speechSynthesis.getVoices(); } catch (e) { /* prompt the browser to load voices */ }
  }
});

// The engine + voice pickers depend on the language — refresh them on change.
const languageSelectEl = document.getElementById('languageSelect');
if (languageSelectEl) {
  languageSelectEl.addEventListener('change', updateVoiceEngineForLanguage);
}

// Chat functionality now handled by new dashboard interface

const includeImageCheckbox = document.getElementById('toggleImagePrompt');
const languageSelect = document.getElementById('languageSelect');

// Initial setup for new dashboard
if (includeImageCheckbox) {
  toggleImageButtons(includeImageCheckbox.checked);
  includeImageCheckbox.addEventListener('change', () => {
    toggleImageButtons(includeImageCheckbox.checked);
  });
}

function toggleImageButtons(enabled) {
  // Disabled until a model is fully loaded, in LLM-only mode, or when the image
  // toggle is off.
  const shouldDisable = (typeof modelReady === 'function' && !modelReady()) || isLlmOnlyMode() || !enabled;

  if (snapChatButton) {
    snapChatButton.disabled = shouldDisable;
  }
  if (uploadButton) {
    uploadButton.disabled = shouldDisable;
  }

  // Handle camera section display mode
  toggleCameraDisplay(enabled);

  // Tooltip logic
  let tooltipMsg;
  if (isLlmOnlyMode()) {
    tooltipMsg = 'Vision features disabled - LLM-only mode';
  } else if (!enabled) {
    tooltipMsg = 'To enable, check "Include image in the prompt" in settings.';
  }

  if (tooltipMsg) {
    if (snapChatButton) snapChatButton.title = tooltipMsg;
    if (uploadButton) uploadButton.title = tooltipMsg;
  } else {
    if (snapChatButton) snapChatButton.removeAttribute('title');
    if (uploadButton) uploadButton.removeAttribute('title');
  }
}

function toggleCameraDisplay(imageEnabled) {
  const cameraSection = document.getElementById('cameraSection');
  const imageOverlay = document.getElementById('imageOverlay');
  const cameraPreview = document.getElementById('cameraPreview');
  const topCenterIcon = document.getElementById('topCenterIcon');

  if (!cameraSection) return;

  if (imageEnabled && !isLlmOnlyMode()) {
    // Show camera feed mode
    cameraSection.classList.remove('logo-mode');
    // Use regular logo (small overlay)
    if (topCenterIcon) {
      topCenterIcon.src = 'static/icons/logo.svg';
    }
    // Restore uploaded image if it exists
    if (imageOverlay && imageOverlay.src) {
      imageOverlay.style.display = 'block';
      cameraPreview.style.display = 'none';
    }
  } else {
    // Show logo mode - hide both camera and uploaded images
    cameraSection.classList.add('logo-mode');

    // Update logo based on current theme
    updateLogoForTheme();

    if (imageOverlay) {
      imageOverlay.style.display = 'none';
    }
    // Reset camera preview display for logo mode
    if (cameraPreview) {
      cameraPreview.style.display = '';
    }
  }
}

document.getElementById("uploadToRagButton").addEventListener("click", async () => {
  if (!isRagEnabled()) return;

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".md";
  fileInput.click();

  fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const messageBox = document.getElementById("settingsMessage");
    messageBox.textContent = "⏳ Creating RAG database from Markdown...";

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/upload-to-rag", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        messageBox.textContent = `❌ Server error: ${response.statusText}`;
        clearMessageLater();
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) {
          buffer += decoder.decode(value, { stream: true });

          let lines = buffer.split("\n");
          buffer = lines.pop();  // save incomplete line

          for (const line of lines) {
            if (line.trim()) {
              messageBox.textContent = line.trim();
            }
          }
        }
      }
    } catch (err) {
      messageBox.textContent = `❌ Error: ${err.message}`;
    }

    clearMessageLater();
  };
});


function clearMessageLater() {
  setTimeout(() => {
    const box = document.getElementById("settingsMessage");
    if (box) box.textContent = "";
  }, 5000);
}

// Shared: POST to a RAG endpoint and stream its text/plain progress into the
// settings message box. Used by "Reset to Default" and "Clear RAG DB".
async function streamRagAction(url, opts) {
  opts = opts || {};
  if (!isRagEnabled()) return;
  if (opts.confirm && !window.confirm(opts.confirm)) return;
  const messageBox = document.getElementById("settingsMessage");
  if (messageBox) messageBox.textContent = "⏳ Working...";
  try {
    const response = await fetch(url, { method: "POST" });
    if (!response.ok) {
      if (messageBox) messageBox.textContent = `❌ Server error: ${response.statusText}`;
      clearMessageLater();
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (line.trim() && messageBox) messageBox.textContent = line.trim();
        }
      }
    }
  } catch (err) {
    if (messageBox) messageBox.textContent = `❌ Error: ${err.message}`;
  }
  // The RAG service was restarted (reset) or stopped (clear) — refresh health.
  if (typeof initializeRagHealth === "function") setTimeout(() => initializeRagHealth(), 1500);
  clearMessageLater();
}

const _resetRagBtn = document.getElementById("resetRagButton");
if (_resetRagBtn) _resetRagBtn.addEventListener("click", () =>
  streamRagAction("/reset-rag", {
    confirm: "Reset the RAG database to the bundled default document? This replaces the current RAG contents."
  }));

const _clearRagBtn = document.getElementById("clearRagButton");
if (_clearRagBtn) _clearRagBtn.addEventListener("click", () =>
  streamRagAction("/clear-rag", {
    confirm: "Clear the RAG database? This removes all ingested documents — you can reset to default or upload again later."
  }));

document.getElementById("importRagDatabaseButton").addEventListener("click", async () => {
  if (!isRagEnabled()) return;

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".db";
  fileInput.click();

  fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const messageBox = document.getElementById("settingsMessage");
    messageBox.textContent = ""; // Clear previous messages

    try {
      const formData = new FormData();
      formData.append("dbfile", file);

      const response = await fetch("/import-rag-db", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        messageBox.textContent = `❌ Server error: ${response.statusText}`;
        setTimeout(() => { messageBox.textContent = ""; }, 5000);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (line.trim()) {
            messageBox.textContent = line.trim();
          }
        }
      }
    } catch (err) {
      messageBox.textContent = `❌ Error: ${err.message}`;
    }

    setTimeout(() => {
      messageBox.textContent = "";
    }, 5000);
  };
});

// Wire language-aware voice selection change
(function setupVoiceSelection() {
  const voiceSelectEl = document.getElementById('voiceSelect');
  const activateButton = document.getElementById('voiceActivateButton');
  if (!voiceSelectEl) return;

  // Controls to disable during voice switch
  const controlIds = [
    'sendButton',
    'recordButton',
    'snapChatButton',
    'uploadButton'
  ];
  function setControlsDisabled(disabled) {
    controlIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        if (!disabled) {
          // When re-enabling, respect image prompt checkbox state
          const imageCheckbox = document.getElementById('toggleImagePrompt');
          if ((id === 'snapChatButton' || id === 'uploadButton') && imageCheckbox && !imageCheckbox.checked) {
            el.disabled = true; // Keep disabled when image prompt is not included
          } else {
            el.disabled = false;
          }
        } else {
          el.disabled = true;
        }
      }
    });
  }

  // Store previous value for revert
  if (!voiceSelectEl.dataset.prev) {
    voiceSelectEl.dataset.prev = voiceSelectEl.value;
  }

  async function activateSelectedVoice() {
    const chosen = voiceSelectEl.value;
    const label = voiceSelectEl.options[voiceSelectEl.selectedIndex]?.text || chosen;
    const voiceMsg = document.getElementById('voiceStatus');
    if (voiceMsg) { voiceMsg.textContent = `⏳ Switching voice to: ${label}...`; }

    setControlsDisabled(true);
    try {
      const res = await fetch('/voices/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang: getSelectedVoiceLanguage(), voiceId: chosen })
      });
      if (!res.ok) throw new Error('Server rejected voice');
      const data = await res.json();
      localStorage.setItem(getVoiceStorageKey(getSelectedVoiceLanguage()), chosen);
      voiceSelectEl.dataset.prev = chosen;
      _currentTtsEngine = data.engine || 'piper-tts';
      // brief confirmation: write into voiceStatus if present
      if (voiceMsg) {
        voiceMsg.textContent = `✅ Voice switched to: ${label}`;
        setTimeout(() => { if (voiceMsg.textContent.startsWith('✅ Voice')) voiceMsg.textContent = ''; }, 2000);
      }
    } catch (e) {
      // revert
      const prev = voiceSelectEl.dataset.prev;
      if (prev) voiceSelectEl.value = prev;
      if (voiceMsg) {
        voiceMsg.textContent = `❌ Failed to switch voice.`;
        setTimeout(() => { if (voiceMsg.textContent.startsWith('❌ Failed')) voiceMsg.textContent = ''; }, 3000);
      }
    } finally {
      setControlsDisabled(false);
      await updateVoiceEngineForLanguage();
    }
  }

  voiceSelectEl.addEventListener('change', activateSelectedVoice);
  if (activateButton) activateButton.addEventListener('click', activateSelectedVoice);
})();

// Theme Management Functions
function getStoredTheme() {
  return localStorage.getItem('theme') || 'dark';
}

function setStoredTheme(theme) {
  localStorage.setItem('theme', theme);
}

function updateLogoForTheme() {
  const topCenterIcon = document.getElementById('topCenterIcon');
  const toggleImagePrompt = document.getElementById('toggleImagePrompt');

  if (topCenterIcon && toggleImagePrompt && !toggleImagePrompt.checked && !isLlmOnlyMode()) {
    // Only update logo if image is disabled and not in LLM-only mode
    const currentTheme = document.documentElement.getAttribute('data-theme');
    if (currentTheme === 'light') {
      topCenterIcon.src = 'static/icons/logo_dark.png';
    } else {
      topCenterIcon.src = 'static/icons/logo_bright.png';
    }
  }
}

function applyTheme(theme) {
  const root = document.documentElement;
  // Fade every element's colors together for the duration of the switch — but
  // not on the initial page-load apply (no fade-in on load).
  if (applyTheme._init) {
    root.classList.add('theme-anim');
    clearTimeout(applyTheme._t);
    applyTheme._t = setTimeout(() => root.classList.remove('theme-anim'), 420);
  }
  applyTheme._init = true;
  root.setAttribute('data-theme', theme);
  updateThemeIcon(theme);
  updateLogoForTheme();
  applyThemedLogos(theme);
}

// Swap SiMa logos to the variant that reads on the current background:
// bright logo on a dark theme, dark logo on a light theme.
function applyThemedLogos(theme) {
  const src = (theme === 'light')
    ? 'static/icons/logo_dark.png'
    : 'static/icons/logo_bright.png';
  document.querySelectorAll('.js-themed-logo').forEach((img) => {
    if (!img.src.endsWith(src)) img.src = src;
  });
}

function updateThemeIcon(theme) {
  if (themeIcon) {
    themeIcon.textContent = theme === 'light' ? '☀️' : '🌙';
  }
}

function toggleTheme() {
  const currentTheme = getStoredTheme();
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';

  setStoredTheme(newTheme);
  applyTheme(newTheme);

  console.log(`🎨 Theme switched to: ${newTheme}`);
}

// Initialize theme on page load
function initializeTheme() {
  const savedTheme = getStoredTheme();
  applyTheme(savedTheme);
  console.log(`🎨 Theme initialized: ${savedTheme}`);
}

// Call theme initialization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initializeTheme();
  applySavedFont();

  // Clear chat history on page refresh
  fetch('/clear-history', { method: 'POST' })
    .catch(error => console.log('Failed to clear history on page load:', error));
});

// Image Modal Functions
function openImageModal(imageSrc) {
  const modal = document.getElementById('imageModal');
  const modalImage = document.getElementById('modalImage');

  modalImage.src = imageSrc;
  modal.style.display = 'flex';

  // Add click-outside-to-close functionality
  modal.addEventListener('click', closeImageModal);

  // Prevent closing when clicking on the image itself
  modalImage.addEventListener('click', (e) => {
    e.stopPropagation();
  });
}

function closeImageModal() {
  const modal = document.getElementById('imageModal');
  modal.style.display = 'none';

  // Remove event listeners to prevent memory leaks
  modal.removeEventListener('click', closeImageModal);
}

/* =======================================================================
 * Neat GenAI Studio: Markdown rendering, live model management, Hugging Face
 * downloads, and font customization. All assets are served locally so the
 * UI runs fully offline on the board.
 * ===================================================================== */

// ---- Markdown rendering ------------------------------------------------

let _markdownReady = false;
let _purifyHookAdded = false;
function ensureMarkdownConfigured() {
  // Open every rendered link (model responses AND model cards) in a new tab.
  // A DOMPurify hook is the single choke point both render paths pass through.
  if (!_purifyHookAdded && window.DOMPurify && typeof window.DOMPurify.addHook === 'function') {
    window.DOMPurify.addHook('afterSanitizeAttributes', (node) => {
      if (node.tagName === 'A' && node.getAttribute('href')) {
        node.setAttribute('target', '_blank');
        node.setAttribute('rel', 'noopener noreferrer');
      }
    });
    _purifyHookAdded = true;
  }
  if (_markdownReady) return;
  if (window.marked && typeof window.marked.setOptions === 'function') {
    window.marked.setOptions({ gfm: true, breaks: true });
    // Offline LaTeX math via KaTeX. marked-katex-extension handles $…$ / $$…$$
    // (with currency false-positive avoidance, before markdown can mangle the
    // math); add extensions for \(…\) and \[…\] which models also emit, plus one
    // for $…$/$$…$$ flush against a bracket — e.g. ($x^2$) — which the currency
    // guard rejects because it wants whitespace after the closing delimiter.
    try {
      if (typeof window.marked.use === 'function' && window.katex) {
        if (window.markedKatex) {
          window.marked.use(window.markedKatex({ throwOnError: false, output: 'html' }));
        }
        window.marked.use({ extensions: [_katexDollarBracket(), _katexParenInline(), _katexBracketBlock()] });
      }
    } catch (e) { /* math is optional — fall back to plain markdown */ }
    _markdownReady = true;
  }
}

// KaTeX renderer for a math token; falls back to the raw source on error.
function _renderKatex(text, displayMode) {
  try {
    return window.katex.renderToString(text, { throwOnError: false, output: 'html', displayMode });
  } catch (e) {
    return displayMode ? '\\[' + text + '\\]' : '\\(' + text + '\\)';
  }
}
// Inline \( … \) math.
function _katexParenInline() {
  return {
    name: 'katexParenInline', level: 'inline',
    start(src) { const i = src.indexOf('\\('); return i < 0 ? undefined : i; },
    tokenizer(src) {
      const m = /^\\\(([\s\S]+?)\\\)/.exec(src);
      if (m) return { type: 'katexParenInline', raw: m[0], text: m[1] };
    },
    renderer(token) { return _renderKatex(token.text, false); },
  };
}
// Display \[ … \] math (block level).
function _katexBracketBlock() {
  return {
    name: 'katexBracketBlock', level: 'block',
    start(src) { const i = src.indexOf('\\['); return i < 0 ? undefined : i; },
    tokenizer(src) {
      const m = /^\\\[([\s\S]+?)\\\]/.exec(src);
      if (m) return { type: 'katexBracketBlock', raw: m[0], text: m[1] };
    },
    renderer(token) { return _renderKatex(token.text, true); },
  };
}
// Inline $ … $ / $$ … $$ math flush against an opening ( or { bracket, e.g.
// "($x^2$)". marked-katex-extension's currency guard skips these because it
// requires whitespace/punctuation after the closing delimiter. Restricted to
// ( and { (not [) so it never swallows a Markdown link's [text].
function _katexDollarBracket() {
  const RE = /^([({])(\$\$?)([\s\S]*?[^\s$])\2/;
  return {
    name: 'katexDollarBracket', level: 'inline',
    start(src) { const m = /[({]\$/.exec(src); return m ? m.index : undefined; },
    tokenizer(src) {
      const m = RE.exec(src);
      if (m) return {
        type: 'katexDollarBracket', raw: m[1] + m[2] + m[3] + m[2],
        bracket: m[1], text: m[3], display: m[2] === '$$',
      };
    },
    renderer(token) { return token.bracket + _renderKatex(token.text, token.display); },
  };
}

function renderMarkdownStreaming(el, text) {
  if (!el) return;
  el._raw = text || '';
  if (window.marked && window.DOMPurify) {
    ensureMarkdownConfigured();
    try {
      const html = window.marked.parse(el._raw);
      el.innerHTML = window.DOMPurify.sanitize(html);
      return;
    } catch (err) {
      /* fall through to plain text */
    }
  }
  el.textContent = el._raw;
}

// Render + highlight + copy buttons (used for non-streaming messages).
function renderMarkdownInto(el, text) {
  renderMarkdownStreaming(el, text);
  enhanceCodeBlocks(el);
}

// Accumulate a streamed chunk and re-render at most once per animation frame.
function appendMarkdownChunk(container, chunk) {
  if (!container) return;
  container._raw = (container._raw || '') + chunk;
  if (container._renderScheduled) return;
  container._renderScheduled = true;
  container._renderRaf = requestAnimationFrame(() => {
    container._renderScheduled = false;
    renderMarkdownStreaming(container, container._raw);
    // The re-render recreated text nodes — reattach the TTS highlight if active.
    if (_ttsHi.container === container) applyTtsHighlight();
    scrollChatToBottom();
  });
}

// Cancel a scheduled throttle render so it can't fire after the final render
// (which would drop syntax highlighting + code-copy buttons).
function cancelPendingRender(container) {
  if (container && container._renderScheduled) {
    try { cancelAnimationFrame(container._renderRaf); } catch (e) { /* ignore */ }
    container._renderScheduled = false;
  }
}

// Like appendMarkdownChunk but SETS the full raw text (used when the text is
// re-derived each delta, e.g. the answer with the <think> block stripped out).
function setMarkdownThrottled(container, fullText) {
  if (!container) return;
  container._raw = fullText || '';
  if (container._renderScheduled) return;
  container._renderScheduled = true;
  container._renderRaf = requestAnimationFrame(() => {
    container._renderScheduled = false;
    renderMarkdownStreaming(container, container._raw || '');
    if (_ttsHi.container === container) applyTtsHighlight();
    scrollChatToBottom();
  });
}

// ---- Thinking (reasoning) rendering ------------------------------------------
// Reasoning models stream their chain-of-thought inline as <think>…</think>
// before the answer. Split it out so the thinking shows in a collapsible block
// above the answer, and the two are counted separately.
function splitThinking(raw) {
  raw = raw || '';
  const openIdx = raw.indexOf('<think>');
  if (openIdx !== -1) {                       // explicit <think>…</think>
    const pre = raw.slice(0, openIdx);
    const afterOpen = raw.slice(openIdx + 7);
    const c = afterOpen.indexOf('</think>');
    if (c === -1) return { thinking: afterOpen, answer: pre, present: true, closed: false };
    return { thinking: afterOpen.slice(0, c), answer: pre + afterOpen.slice(c + 8), present: true, closed: true };
  }
  const closeIdx = raw.indexOf('</think>');   // template pre-filled <think>; only the close is emitted
  if (closeIdx !== -1) {
    return { thinking: raw.slice(0, closeIdx), answer: raw.slice(closeIdx + 8), present: true, closed: true };
  }
  return { thinking: '', answer: raw, present: false, closed: true };
}

function ensureThinkBlock(messageEl) {
  if (messageEl._thinkBlock) return messageEl._thinkBlock;
  const block = document.createElement('details');
  block.className = 'think-block';
  block.open = true;
  const summary = document.createElement('summary');
  summary.className = 'think-summary';
  const label = document.createElement('span');
  label.className = 'think-label';
  label.textContent = 'Thinking';
  const toks = document.createElement('span');
  toks.className = 'think-tokens';
  summary.appendChild(label);
  summary.appendChild(toks);
  const body = document.createElement('div');
  body.className = 'think-text';   // NOT .message-text, so the answer lookup stays unambiguous
  block.appendChild(summary);
  block.appendChild(body);
  const answerEl = messageEl.querySelector('.message-text');
  if (answerEl) messageEl.insertBefore(block, answerEl);
  else messageEl.insertBefore(block, messageEl.firstChild);
  messageEl._thinkBlock = block;
  block._thinkText = body;
  block._thinkTokensEl = toks;
  return block;
}

function enhanceCodeBlocks(el) {
  if (!el) return;
  el.querySelectorAll('pre > code').forEach(code => {
    if (window.hljs && code.dataset.highlighted !== 'true') {
      try { window.hljs.highlightElement(code); } catch (e) { /* ignore */ }
      code.dataset.highlighted = 'true';
    }
    const pre = code.parentElement;
    if (pre && !pre.querySelector('.code-copy-btn')) {
      pre.classList.add('code-block');
      const btn = document.createElement('button');
      btn.className = 'code-copy-btn';
      btn.type = 'button';
      btn.textContent = 'Copy';
      btn.addEventListener('click', () => copyTextToClipboard(code.innerText, btn));
      pre.appendChild(btn);
    }
  });
}

// Copy text to the clipboard, falling back to a hidden textarea when the async
// Clipboard API is unavailable (older browsers / insecure contexts); flashes the
// button on success.
function copyTextToClipboard(text, btn) {
  const done = () => flashCopyButton(btn);
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(() => copyFallback(text, done));
  } else {
    copyFallback(text, done);
  }
}

function copyFallback(text, done) {
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-1000px';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    if (done) done();
  } catch (e) { /* ignore */ }
}

function flashCopyButton(btn) {
  if (!btn) return;
  btn.classList.add('copied');
  const iconOnly = btn.dataset.icon === 'true';
  const prev = iconOnly ? null : btn.textContent;
  if (!iconOnly) btn.textContent = 'Copied';
  clearTimeout(btn._copyTimer);
  btn._copyTimer = setTimeout(() => {
    btn.classList.remove('copied');
    if (!iconOnly) btn.textContent = prev;
  }, 1200);
}

// Add a hover "copy response" button to an assistant message (idempotent). It
// copies the raw reply text (Markdown source when available, else the rendered
// text), so users can copy the whole answer, not just individual code blocks.
function addResponseCopyButton(messageDiv) {
  if (!messageDiv || messageDiv.querySelector('.msg-copy-btn')) return;
  const btn = document.createElement('button');
  btn.className = 'msg-copy-btn';
  btn.type = 'button';
  btn.dataset.icon = 'true';
  btn.title = 'Copy response';
  btn.setAttribute('aria-label', 'Copy response');
  btn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
    + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    + '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 012-2h10"/></svg>';
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const textEl = messageDiv.querySelector('.message-text') || messageDiv;
    const text = (textEl && textEl._raw) ? textEl._raw
      : (textEl ? textEl.innerText : messageDiv.innerText);
    copyTextToClipboard(text, btn);
  });
  messageDiv.appendChild(btn);
}

// ---- Live model management --------------------------------------------

let _modelBusy = false;

function controlEnabled() {
  const c = window.SIMA_CONFIG || {};
  return c.controlEnabled === true || c.controlEnabled === 'true';
}

function setModelStatus(text, kind) {
  const el = document.getElementById('modelStatus');
  if (_loadMirror.active) mirrorLoadStatus(text, kind);
  if (!el) return;
  el.textContent = text || '';
  el.className = 'model-status' + (kind ? ' ' + kind : '');
}

// When another surface (e.g. the full-screen Benchmark) needs to show the same
// model-load progress, it turns on a mirror so status + log lines are echoed
// into its own panel without re-streaming the load feed.
const _loadMirror = { active: false, statusEl: null, logEl: null, barEl: null };
function beginLoadMirror(statusId, logId, barId) {
  _loadMirror.active = true;
  _loadMirror.statusEl = document.getElementById(statusId);
  _loadMirror.logEl = document.getElementById(logId);
  _loadMirror.barEl = barId ? document.getElementById(barId) : null;
  if (_loadMirror.logEl) _loadMirror.logEl.textContent = '';
  if (_loadMirror.barEl) _loadMirror.barEl.classList.add('active');
}
function endLoadMirror() {
  if (_loadMirror.barEl) _loadMirror.barEl.classList.remove('active');
  _loadMirror.active = false;
  _loadMirror.statusEl = _loadMirror.logEl = _loadMirror.barEl = null;
}
function mirrorLoadStatus(text, kind) {
  const el = _loadMirror.statusEl;
  if (!el) return;
  el.textContent = text || '';
  el.className = 'bench-load-status' + (kind ? ' ' + kind : '');
}

function typeBadge(type) {
  const t = (type || 'chat').toLowerCase();
  if (t === 'vlm') return 'VLM';
  if (t === 'asr') return 'ASR';
  return 'LLM';
}

// Human-readable byte size, e.g. 3.8 GB / 512 MB.
function fmtBytes(n) {
  n = Number(n);
  if (!Number.isFinite(n) || n <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${units[i]}`;
}

// Human-readable duration, e.g. 1m 20s / 45s.
function fmtDuration(s) {
  s = Math.round(Number(s));
  if (!Number.isFinite(s) || s < 0) return '';
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

async function initStudioModelManager() {
  const row = document.getElementById('chatModelRow');
  if (row) row.style.display = '';   // reveal (CSS gives it the flex-column layout)
  if (!controlEnabled()) {
    // Static mode: models are preloaded server-side and switching is instant, so
    // there is nothing to load/unload — hide those controls and treat a row
    // pick (which writes the hidden <select>) as an immediate switch.
    const progress = document.getElementById('modelLoadProgress');
    if (progress) progress.style.display = 'none';
    populateModelSelect(getConfiguredChatModels().map(name => ({ name, loaded: true, type: 'chat' })));
    const select = document.getElementById('chatModelSelect');
    if (select) {
      select.addEventListener('change', () => {
        _activeChatModel = select.value || '';
        updateActiveModelPill(_activeChatModel || '—');
        if (typeof updateSelectedModelVisionState === 'function') updateSelectedModelVisionState();
      });
    }
    return;
  }
  // Control mode: models load on demand. The unified list's per-row Load/Unload
  // buttons drive everything (wired in renderInstalledList); refresh + search are
  // wired in initHubControls.
  await refreshCatalog();
}

async function refreshCatalog() {
  try {
    const resp = await fetch('/models/catalog');
    const data = await resp.json();
    const catalog = (data && data.catalog) || [];
    // Update capabilities so vision detection works for any catalog model.
    const caps = window.SIMA_CONFIG.chatModelCapabilities || {};
    catalog.forEach(m => {
      if (m.type !== 'asr') {
        caps[m.name] = { supportsVision: !!m.supportsVision, imageSize: m.imageSize || null };
      }
    });
    window.SIMA_CONFIG.chatModelCapabilities = caps;
    populateModelSelect(catalog);
    return catalog;
  } catch (err) {
    console.warn('Failed to load model catalog:', err);
    setModelStatus('Model catalog unavailable', 'error');
    return [];
  }
}

function populateModelSelect(catalog) {
  const select = document.getElementById('chatModelSelect');
  if (!select) return;
  _catalog = catalog || [];
  // Speech models are tracked separately: they never appear in the chat select,
  // and exactly one of them is active at a time.
  const asrModels = _catalog.filter(m => (m.type || 'chat') === 'asr');
  _asrActive = (asrModels.find(m => m.activeAsr) || {}).name || '';
  updateAsrModelIndicator();
  const chatModels = catalog.filter(m => (m.type || 'chat') !== 'asr');
  const previous = select.value;
  while (select.firstChild) select.removeChild(select.firstChild);

  const loaded = chatModels.filter(m => m.loaded).map(m => m.name);
  const anyLoaded = loaded.length > 0;

  // When nothing is loaded, lead with a placeholder so no unloaded model is
  // treated as active — the user picks one, which triggers a load.
  if (!anyLoaded) {
    const ph = document.createElement('option');
    ph.value = '';
    ph.textContent = chatModels.length ? 'Select a model to load…' : 'No models — download one below';
    ph.dataset.loaded = 'false';
    select.appendChild(ph);
  }

  chatModels.forEach(m => {
    const option = document.createElement('option');
    option.value = m.name;
    const size = m.sizeBytes ? `  ·  ${fmtBytes(m.sizeBytes)}` : '';
    const incomplete = m.complete === false;
    const dot = incomplete ? '⚠ ' : (m.loaded ? '● ' : '○ ');
    option.textContent = `${dot}${m.name}  ·  ${typeBadge(m.type)}${size}${incomplete ? '  ·  incomplete' : ''}`;
    option.dataset.loaded = m.loaded ? 'true' : 'false';
    option.dataset.complete = incomplete ? 'false' : 'true';
    if (incomplete && m.incompleteReason) option.dataset.incompleteReason = m.incompleteReason;
    select.appendChild(option);
  });

  const defaultModel = window.SIMA_CONFIG?.defaultChatModel;
  // Preserve the user's browsing selection across refreshes; fall back to the
  // loaded model (or default) when the prior selection is gone.
  let selection = previous;
  if (!chatModels.some(m => m.name === selection)) {
    selection = anyLoaded
      ? (loaded.includes(defaultModel) ? defaultModel : loaded[0])
      : '';
  }
  select.value = selection;

  // Active model = the one actually resident. With the control API only one
  // chat/VLM is loaded at a time; in static mode every model is preloaded so
  // the active one follows the current selection.
  if (controlEnabled()) {
    _activeChatModel = loaded.includes(defaultModel) ? defaultModel : (loaded[0] || '');
  } else {
    _activeChatModel = select.value || defaultModel || (chatModels[0] && chatModels[0].name) || '';
  }

  const total = chatModels.length;
  if (!anyLoaded) {
    setModelStatus(total ? 'No model loaded — pick one and press Load' : 'No models yet — download one from Hugging Face', 'muted');
  } else {
    setModelStatus(`${loaded.length} loaded · ${total} in catalog`, 'muted');
  }
  updateActiveModelPill(_activeChatModel || '—');
  updateManageButtons();
  if (typeof updateSelectedModelVisionState === 'function') updateSelectedModelVisionState();
}

// ---- Unified, searchable model list -----------------------------------
// The visible Models tab is a single searchable list: installed models (from the
// catalog) plus, when online, models available on Hugging Face. The hidden
// <select id="chatModelSelect"> stays the source of truth for the rest of the app.
let _catalog = [];
let _pendingLoad = '';    // name of the model currently loading (for the row label)
let _resetting = false;   // an accelerator reset + server relaunch is in flight
let _loadTicker = null;   // client-side load countdown (see startLoadTicker)
let _lastServerLoadUpdate = 0;   // when the server last reported real progress
let _asrActive = '';     // ASR model serving transcriptions
let _asrPending = '';    // ASR model mid-switch (for the row label)

function escHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// Static mode (no control API): clicking a model row switches to it instantly.
function selectInstalledModel(name) {
  if (controlEnabled()) return;   // control mode uses the per-row Load button
  const select = document.getElementById('chatModelSelect');
  if (select) { select.value = name; select.dispatchEvent(new Event('change')); }
  renderInstalledList();
}

// Fill the Models-tab family filter from the installed models (rebuilt only when
// the set of families actually changes, so it doesn't churn on every render).
function populateModelFamilyFilter(models) {
  const sel = document.getElementById('modelFilterFamily');
  if (!sel) return;
  const fams = Array.from(new Set(models.map(m => hubModelFamily(m.name)))).sort((a, b) => a.localeCompare(b));
  const want = fams.join('\n');
  if (sel.dataset.fams === want) return;
  sel.dataset.fams = want;
  const cur = sel.value;
  sel.innerHTML = '<option value="">All families</option>'
    + fams.map(f => `<option value="${escHtml(f)}">${escHtml(f)}</option>`).join('');
  if (fams.includes(cur)) sel.value = cur;
}

// Render the installed models into #modelInstalledList, honouring the search box
// + the type / size / family / sort filters. Called on catalog or state changes.
function renderInstalledList() {
  const list = document.getElementById('modelInstalledList');
  if (!list) return;
  const text = (document.getElementById('modelSearchInput')?.value || '').trim().toLowerCase();
  const ft = document.getElementById('modelFilterType')?.value || '';
  const fp = document.getElementById('modelFilterParams')?.value || '';
  const ff = document.getElementById('modelFilterFamily')?.value || '';
  const sort = document.getElementById('modelSortBy')?.value || 'loaded';
  const models = _catalog.filter(m => (m.type || 'chat') !== 'asr');
  populateModelFamilyFilter(models);
  const filtered = models.filter(m => {
    if (text && !m.name.toLowerCase().includes(text)) return false;
    if (ft && (m.supportsVision ? 'VLM' : 'LLM') !== ft) return false;
    if (fp && hubParamsBucket(hubModelParams(m.name)) !== fp) return false;
    if (ff && hubModelFamily(m.name) !== ff) return false;
    return true;
  });
  const pv = (n) => hubModelParams(n);
  filtered.sort((a, b) => {
    switch (sort) {
      case 'name': return a.name.localeCompare(b.name);
      case 'size-desc': return (b.sizeBytes || 0) - (a.sizeBytes || 0);
      case 'size-asc': return (a.sizeBytes || Infinity) - (b.sizeBytes || Infinity);
      case 'params-desc': return (pv(b.name) || 0) - (pv(a.name) || 0);
      case 'params-asc': return (pv(a.name) == null ? Infinity : pv(a.name)) - (pv(b.name) == null ? Infinity : pv(b.name));
      default: return (b.loaded ? 1 : 0) - (a.loaded ? 1 : 0) || a.name.localeCompare(b.name);
    }
  });

  const countEl = document.getElementById('modelInstalledCount');
  if (countEl) countEl.textContent = models.length ? `${filtered.length} of ${models.length}` : '';

  list.innerHTML = '';
  if (!models.length) {
    list.innerHTML = `<div class="hub-note">Nothing downloaded yet${_hubEnabled ? ' — get one from the “Add Model” tab.' : '.'}</div>`;
    return;
  }
  if (!filtered.length) {
    list.innerHTML = '<div class="hub-note">No downloaded models match the search.</div>';
    return;
  }

  const control = controlEnabled();
  const busy = serverBusy();
  const activeName = control ? _activeChatModel : getSelectedChatModel();
  filtered.forEach(m => {
    const incomplete = m.complete === false;
    const isActive = !!m.name && m.name === activeName;
    const row = document.createElement('div');
    row.className = 'hub-result model-row' + (isActive ? ' is-active' : '');

    const meta = document.createElement('div');
    meta.className = 'hub-result-meta';
    const t = m.supportsVision ? 'VLM' : 'LLM';
    const size = m.sizeBytes ? fmtBytes(m.sizeBytes) : '';
    const stateCls = incomplete ? 'is-incomplete' : (m.loaded ? 'is-loaded' : '');
    // "downloaded" (on disk, ready to load) vs "loaded" (in memory now) — never
    // "available", which reads like "available to download".
    const stateTxt = incomplete ? '⚠ incomplete' : (m.loaded ? '● loaded' : '○ downloaded');
    const badges = `<span class="hub-badge hub-badge-${t.toLowerCase()}">${t}</span>`
      + (size ? `<span class="hub-badge">${size}</span>` : '')
      + `<span class="hub-badge model-state ${stateCls}">${stateTxt}</span>`;
    meta.innerHTML = `<span class="hub-repo">${escHtml(m.name)}</span><span class="hub-badges">${badges}</span>`;
    row.appendChild(meta);
    if (!control) row.addEventListener('click', (e) => { if (!e.target.closest('button')) selectInstalledModel(m.name); });

    const info = document.createElement('button');
    info.className = 'hub-info'; info.type = 'button'; info.textContent = 'ℹ';
    info.title = `Model card & metadata for ${m.name}`;
    info.addEventListener('click', (e) => { e.stopPropagation(); showModelCard(m.name); });
    row.appendChild(info);

    if (control) {
      const del = document.createElement('button');
      del.className = 'hub-info hub-danger'; del.type = 'button'; del.textContent = '🗑';
      del.title = `Delete ${m.name} from disk`;
      del.disabled = busy;
      del.addEventListener('click', (e) => { e.stopPropagation(); deleteModel(m.name); });
      row.appendChild(del);

      const btn = document.createElement('button');
      btn.className = 'setting-button model-action'; btn.type = 'button';
      if (m.loaded) {
        btn.textContent = 'Unload'; btn.classList.add('model-unload'); btn.disabled = busy;
        btn.addEventListener('click', (e) => { e.stopPropagation(); unloadModel(m.name); });
      } else if (incomplete) {
        btn.textContent = 'Incomplete'; btn.disabled = true;
        btn.title = `${m.incompleteReason || 'Weights are incomplete'} — re-download from Hugging Face below.`;
      } else if (_modelBusy && m.name === _pendingLoad) {
        btn.textContent = 'Loading…'; btn.disabled = true;
      } else {
        btn.textContent = 'Load'; btn.classList.add('model-load'); btn.disabled = busy;
        btn.addEventListener('click', (e) => { e.stopPropagation(); loadModelAndActivate(m.name); });
      }
      row.appendChild(btn);
    } else {
      const btn = document.createElement('button');
      btn.className = 'setting-button model-action'; btn.type = 'button';
      btn.textContent = isActive ? 'Active' : 'Use';
      btn.disabled = isActive;
      btn.addEventListener('click', (e) => { e.stopPropagation(); selectInstalledModel(m.name); });
      row.appendChild(btn);
    }
    list.appendChild(row);
  });
}

// Show how much room is left on the NVMe (the filesystem holding the catalog),
// so the user can judge whether a download will fit.
function renderDiskInfo(disk) {
  const el = document.getElementById('modelDiskInfo');
  if (!el) return;
  if (disk && typeof disk.freeBytes === 'number') {
    const total = disk.totalBytes ? ` of ${fmtBytes(disk.totalBytes)}` : '';
    el.textContent = `NVMe storage: ${fmtBytes(disk.freeBytes)} free${total}`;
    el.style.display = '';
  } else {
    el.style.display = 'none';
  }
}

// Re-read free space after a download or delete changes it.
async function refreshDiskInfo() {
  try {
    const r = await fetch('/models/status', { cache: 'no-store' });
    const d = await r.json();
    renderDiskInfo(d && d.disk);
  } catch (e) { /* leave the current value */ }
}

// The header pill persists in the chat view (unlike the home-screen indicator,
// which is hidden once messages appear), so the active model stays visible while
// chatting. Show it when a model is resident; hide it when none.
function updateActiveModelPill() {
  const pill = document.getElementById('headerModelPill');
  const nameEl = document.getElementById('headerModelName');
  const model = getSelectedChatModel();
  if (pill && nameEl) {
    if (model) {
      nameEl.textContent = model;
      pill.style.display = '';
    } else {
      pill.style.display = 'none';
    }
  }
  updateHomeModelIndicator();
}

// Show the active model on the home (empty) screen so it is clear which model
// will answer — or invite the user to load one when none is resident.
function updateHomeModelIndicator() {
  const box = document.getElementById('homeModelIndicator');
  const nameEl = document.getElementById('homeModelName');
  if (!box || !nameEl) return;
  const model = getSelectedChatModel();
  if (model) {
    const vision = typeof selectedChatModelSupportsVision === 'function' && selectedChatModelSupportsVision();
    nameEl.textContent = model + (vision ? '  ·  vision' : '');
    box.classList.add('loaded');
    box.classList.remove('empty');
    box.title = 'Active model — click to change';
  } else {
    nameEl.textContent = 'No model loaded — choose one';
    box.classList.add('empty');
    box.classList.remove('loaded');
    box.title = 'Open Settings to load a model';
  }
}

// Refresh the per-row actions in the model list (Load / Unload / active state)
// and composer whenever model state changes.
function updateManageButtons() {
  // Deliberately NOT disabled by _modelBusy: a wedged load holds that flag for
  // the life of the request (up to 600s), which is exactly when this recovery
  // action is needed. reset_mla() bypasses the server's op-lock for the same
  // reason. Only a reset already in flight disables it.
  const mlaReset = document.getElementById('mlaResetButton');
  if (mlaReset) mlaReset.disabled = _resetting;
  renderInstalledList();
  renderAsrList();
  updateComposerEnabled();
}

// A model is usable for chat only once it is FULLY resident (not mid-load and
// not mid-reset). The composer is locked until then.
function serverBusy() {
  // Any state in which the model server cannot service a request: a model
  // operation in flight, or a reset during which the old server is exiting and
  // the replacement is not yet up.
  return _modelBusy || _resetting;
}

function modelReady() {
  return !!getSelectedChatModel() && !serverBusy();
}

// Enable/disable the chat input interface based on model readiness.
function updateComposerEnabled() {
  const ready = modelReady();
  const input = document.getElementById('messageInput');
  const send = document.getElementById('sendButton');
  const composer = document.getElementById('chatInput');
  if (input) {
    input.disabled = !ready;
    input.placeholder = ready ? 'Message'
      : (_modelBusy ? 'Loading model — please wait…'
        : 'Load a model in Settings to start chatting…');
  }
  if (send) send.disabled = !ready;
  if (typeof recordButton !== 'undefined' && recordButton) recordButton.disabled = !ready;
  if (composer) composer.classList.toggle('composer-locked', !ready);
  // Snap/upload additionally require a vision model. When not ready, force them
  // off here (the load-start case); when ready, updateSelectedModelVisionState ->
  // toggleImageButtons re-evaluates them (it also checks modelReady()).
  if (!ready) {
    if (typeof snapChatButton !== 'undefined' && snapChatButton) snapChatButton.disabled = true;
    if (typeof uploadButton !== 'undefined' && uploadButton) uploadButton.disabled = true;
  }
}

function initModelManage() {
  // Info / delete / load / unload are per-row buttons in the list now.
  const sel = document.getElementById('chatModelSelect');
  if (sel) sel.addEventListener('change', updateManageButtons);

  const mlaReset = document.getElementById('mlaResetButton');
  if (mlaReset) mlaReset.addEventListener('click', () => resetMla());

  const retry = document.getElementById('modelLoadRetry');
  const viewLogs = document.getElementById('modelLoadErrorLogs');
  if (retry) retry.addEventListener('click', () => {
    const box = document.getElementById('modelLoadError');
    const target = (box && box.dataset.model) || (sel && sel.value);
    clearModelError();
    if (!target) return;
    // An ASR retry must go through the switch path so the UI process learns
    // the new name; a plain load would leave transcription on the old model.
    const entry = _catalog.find(m => m.name === target);
    if (entry && (entry.type || 'chat') === 'asr') switchAsrModel(target);
    else loadModelAndActivate(target);
  });
  if (viewLogs) viewLogs.addEventListener('click', () => {
    const det = document.getElementById('modelLogDetails');
    if (det) { det.open = true; det.scrollIntoView({ block: 'nearest' }); }
  });
  updateManageButtons();
}

// Explicitly unload the resident chat/VLM model, freeing the accelerator.
async function unloadModel(name) {
  if (!name || _modelBusy) return;
  if (!window.confirm(`Unload "${name}" from the accelerator? You can load it again anytime.`)) return;
  _modelBusy = true;
  updateManageButtons();
  setModelStatus(`Unloading ${name}…`, 'loading');
  try {
    const resp = await fetch('/models/unload', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || 'unload failed');
    setModelStatus(`Unloaded ${name}`, 'muted');
  } catch (err) {
    setModelStatus(`Failed to unload ${name}: ${err.message}`, 'error');
  } finally {
    _modelBusy = false;
    await refreshCatalog();
    if (typeof updateSelectedModelVisionState === 'function') updateSelectedModelVisionState();
  }
}

async function deleteModel(name, extraWarning) {
  if (!name) return;
  if (!window.confirm(`Delete "${name}" from disk? This removes the model weights `
    + `and cannot be undone.${extraWarning || ''}`)) return;
  setModelStatus(`Deleting ${name}…`, 'loading');
  try {
    const resp = await fetch('/models/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || 'delete failed');
    setModelStatus(`Deleted ${name}`, 'muted');
  } catch (err) {
    setModelStatus(`Failed to delete ${name}: ${err.message}`, 'error');
  } finally {
    await refreshCatalog();
    refreshDiskInfo();    // freeing weights returns space to the NVMe
  }
}

// Name the model behind the "Heard you" metrics, so the numbers below it are
// attributable — they change meaningfully when you switch speech models.
function updateAsrModelIndicator() {
  const el = document.getElementById('asrModelName');
  if (!el) return;
  // In control mode the catalog is authoritative — an empty _asrActive means the
  // server reports none active, and naming the configured model anyway would
  // credit transcripts to a model that is not running. The configured name is
  // only a stand-in for static mode, which has no catalog to consult.
  const name = _asrActive || (controlEnabled() ? '' : (window.SIMA_CONFIG?.asrModelName || ''));
  el.textContent = name;
  el.title = name ? `Transcribed by ${name}` : '';
}

// Drive the load bar from the browser's own clock.
//
// The server cannot report progress while a load is in flight: add_model() does
// the bulk MLA transfer in native code without releasing the GIL, so the whole
// model-server process — control API included — stops answering for the entire
// load (measured: 6s timeouts, zero bytes, for a 20s load). Polling therefore
// yields nothing exactly when there is something to show. So take the estimate
// from the catalog BEFORE starting, and count down locally.
//
// Server-side polling still runs and wins whenever it does answer (a short load,
// or an ASR warm-up, where the process is responsive); this only fills the gap.
function startLoadTicker(name, estimateS, stagesTotal) {
  stopLoadTicker();
  _lastServerLoadUpdate = 0;
  const started = Date.now();
  const est = (typeof estimateS === 'number' && estimateS > 0) ? estimateS : null;
  const tick = () => {
    // Server-reported progress is authoritative; only fill in while it is silent.
    if (Date.now() - _lastServerLoadUpdate < 2000) return;
    const elapsed = (Date.now() - started) / 1000;
    // Hold at 99: the load is finished when the request returns, not when the
    // estimate runs out, and a bar that sits at 100% while still working lies.
    const pct = est ? Math.min(99, Math.floor(elapsed / est * 100)) : null;
    const parts = [];
    if (pct != null) parts.push(`${pct}%`);
    if (stagesTotal) parts.push(`${stagesTotal} stages`);
    parts.push(fmtDuration(Math.round(elapsed)));
    if (est) {
      const remain = Math.max(0, est - elapsed);
      parts.push(remain > 0 ? `~${fmtDuration(Math.round(remain))} left` : 'finishing…');
    }
    setModelStatus(`Loading ${name} · ${parts.join(' · ')}`, 'loading');
    setModelLoadBar(pct != null ? pct : 'active');
  };
  tick();
  _loadTicker = setInterval(tick, 250);
}

function stopLoadTicker() {
  if (_loadTicker) { clearInterval(_loadTicker); _loadTicker = null; }
}

// The catalog carries a per-model estimate and stage count; both may be absent
// for a model the board has never sized.
function modelLoadHints(name) {
  const m = _catalog.find(x => x.name === name) || {};
  return { est: m.estimatedLoadS, stages: m.stagesTotal };
}

// Reset the accelerator: asks the model server to exit with the sentinel code so
// the supervisor (run.sh) restarts the MLA dispatcher — which owns models across
// processes, so killing the server alone does not free them — and relaunches it.
// Explicit only: nothing else in the studio triggers this.
async function resetMla() {
  if (_resetting) return;
  if (!window.confirm('Reset the accelerator (MLA)?\n\nThis unloads all models and briefly '
      + 'restarts the model server — it will be unavailable for a few seconds. '
      + 'In-progress generation will stop.')) return;
  _resetting = true;
  stopLoadPolling();          // the outgoing server's log feed is about to die
  stopLoadTicker();
  clearModelError();
  updateManageButtons();
  setModelStatus('Resetting the accelerator and restarting…', 'loading');
  setModelLoadBar('active');
  try {
    // The server exits ~1.5s after replying, so this may never return — that is
    // the success path, not a failure.
    try { await fetch('/models/reset-mla', { method: 'POST' }); } catch (e) { /* expected */ }
    await waitForServerBack();
  } finally {
    // Always release the lock, even if the wait threw, so the UI cannot get
    // stuck with every action disabled.
    _resetting = false;
    _modelBusy = false;       // a wedged load is gone with the restart
    setModelLoadBar(null);
    clearModelError();        // drop a stale error a concurrent load's 502 raised
    await refreshCatalog();
    updateManageButtons();
    if (typeof updateSelectedModelVisionState === 'function') updateSelectedModelVisionState();
  }
}

// Poll /models/status until the relaunched model server answers.
async function waitForServerBack() {
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const start = Date.now();
  const timeoutMs = 90000;
  await sleep(2500);          // let the old process exit before polling
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch('/models/status', { cache: 'no-store' });
      if (r.ok) {
        const d = await r.json().catch(() => ({}));
        // Back up. Whether the dispatcher itself was reset depends on run.sh
        // having the privileges, so do not over-claim a full accelerator reset.
        if (d && !d.error) { setModelStatus('Model server restarted — ready', 'ready'); return; }
      }
    } catch (e) { /* still down */ }
    await sleep(1500);
  }
  setModelStatus('Reset requested, but the server is slow to return — check run.sh.', 'error');
}

// ---- Speech-to-text (ASR) models --------------------------------------------
// Kept in their own list because they never compete with chat/VLM models for the
// same slot: exactly one ASR model is resident, and picking another evicts it
// without disturbing the loaded chat model. The chat filters (LLM/VLM, parameter
// count, family) are meaningless here, so only the search box applies.
function renderAsrList() {
  const list = document.getElementById('asrModelList');
  if (!list) return;
  const text = (document.getElementById('modelSearchInput')?.value || '').trim().toLowerCase();
  const models = _catalog.filter(m => (m.type || 'chat') === 'asr')
                         .sort((a, b) => a.name.localeCompare(b.name));

  const nameEl = document.getElementById('asrActiveName');
  if (nameEl) nameEl.textContent = _asrActive || (models.length ? 'none active' : '');
  const noteEl = document.getElementById('asrModelNote');
  if (noteEl) noteEl.style.display = models.length > 1 ? '' : 'none';

  list.innerHTML = '';
  if (!models.length) {
    list.innerHTML = `<div class="hub-note">No speech-to-text model${_hubEnabled
      ? ' — download a Whisper model from the “Add Model” tab.' : '.'}</div>`;
    return;
  }
  const filtered = models.filter(m => !text || m.name.toLowerCase().includes(text));
  if (!filtered.length) {
    list.innerHTML = '<div class="hub-note">No speech-to-text models match the search.</div>';
    return;
  }

  const control = controlEnabled();
  const busy = serverBusy();
  filtered.forEach(m => {
    const incomplete = m.complete === false;
    const isActive = !!m.name && m.name === _asrActive;
    const row = document.createElement('div');
    row.className = 'hub-result model-row' + (isActive ? ' is-active' : '');

    const meta = document.createElement('div');
    meta.className = 'hub-result-meta';
    const size = m.sizeBytes ? fmtBytes(m.sizeBytes) : '';
    const stateCls = incomplete ? 'is-incomplete' : (isActive ? 'is-loaded' : '');
    const stateTxt = incomplete ? '⚠ incomplete' : (isActive ? '🎙 active' : '○ downloaded');
    const badges = `<span class="hub-badge hub-badge-asr">${typeBadge('asr')}</span>`
      + (size ? `<span class="hub-badge">${size}</span>` : '')
      + (m.pinned ? '<span class="hub-badge">startup default</span>' : '')
      + `<span class="hub-badge model-state ${stateCls}">${stateTxt}</span>`;
    meta.innerHTML = `<span class="hub-repo">${escHtml(m.name)}</span>`
      + `<span class="hub-badges">${badges}</span>`;
    row.appendChild(meta);

    const info = document.createElement('button');
    info.className = 'hub-info';
    info.type = 'button';
    info.textContent = 'ℹ';
    info.title = `Model card & metadata for ${m.name}`;
    info.addEventListener('click', (e) => { e.stopPropagation(); showModelCard(m.name); });
    row.appendChild(info);

    if (control) {
      // The active model has no delete button — the server refuses it anyway.
      if (!isActive) {
        const del = document.createElement('button');
        del.className = 'hub-info hub-danger';
        del.type = 'button';
        del.textContent = '🗑';
        del.title = `Delete ${m.name} from disk`;
        del.disabled = busy;
        del.addEventListener('click', (e) => {
          e.stopPropagation();
          deleteModel(m.name, m.pinned
            ? ' It is the startup default, so after a restart no speech-to-text '
              + 'model will be active until you pick one.'
            : '');
        });
        row.appendChild(del);
      }

      const btn = document.createElement('button');
      btn.className = 'setting-button model-action';
      btn.type = 'button';
      if (isActive) {
        btn.textContent = 'Active';
        btn.disabled = true;
      } else if (incomplete) {
        btn.textContent = 'Incomplete';
        btn.disabled = true;
        btn.title = `${m.incompleteReason || 'Weights are incomplete'} — re-download from the Add Model tab.`;
      } else if (busy && m.name === _asrPending) {
        btn.textContent = 'Switching…';
        btn.disabled = true;
      } else {
        btn.textContent = 'Use';
        btn.classList.add('model-load');
        btn.disabled = busy;
        btn.addEventListener('click', (e) => { e.stopPropagation(); switchAsrModel(m.name); });
      }
      row.appendChild(btn);
    }
    list.appendChild(row);
  });
}

// Make another ASR model active. Mirrors loadModelAndActivate, minus the chat
// select and minus newChat() — switching how your voice is transcribed is no
// reason to throw away the conversation.
async function switchAsrModel(name) {
  if (!name || _modelBusy || name === _asrActive) return;
  _modelBusy = true;
  _asrPending = name;
  updateManageButtons();
  clearModelError();
  await resetLoadLog();

  const note = _asrActive ? `Unloading ${_asrActive} — ` : '';
  setModelStatus(`${note}Switching speech-to-text to ${name}…`, 'loading');
  setModelLoadBar('active');
  startLoadPolling(name);
  const asrHints = modelLoadHints(name);
  startLoadTicker(name, asrHints.est, asrHints.stages);
  try {
    const resp = await fetch('/models/asr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || 'switch failed');
    _asrActive = data.activeAsr || name;
    const evicted = Array.isArray(data.evicted) ? data.evicted : [];
    const evictedNote = evicted.length ? ` · unloaded ${evicted.join(', ')}` : '';
    const secs = (typeof data.load_seconds === 'number') ? data.load_seconds : null;
    const timeNote = (secs != null && secs > 0) ? ` in ${secs.toFixed(1)}s` : '';
    setModelStatus(`Speech-to-text ready: ${_asrActive}${timeNote}${evictedNote}`, 'ready');
  } catch (err) {
    setModelStatus('Speech-to-text switch failed — see details below', 'error');
    showModelError(name, err.message);
  } finally {
    stopLoadPolling();
    stopLoadTicker();
    _modelBusy = false;
    _asrPending = '';
    await pollLoadLogsOnce(name);
    setModelLoadBar(null);
    await refreshCatalog();   // re-derives _asrActive from the catalog's activeAsr
    updateManageButtons();
  }
}

// ---- Model card + metadata ----
function showModelCard(name) {
  openCardModal(name, '/models/card?name=' + encodeURIComponent(name));
}
function showHubCard(repoId) {
  openCardModal(repoId, '/models/hub/card?repoId=' + encodeURIComponent(repoId));
}
async function openCardModal(title, url) {
  const modal = document.getElementById('modelCardModal');
  const titleEl = document.getElementById('modelCardTitle');
  const metaEl = document.getElementById('modelCardMeta');
  const bodyEl = document.getElementById('modelCardBody');
  if (!modal || !bodyEl) return;
  if (titleEl) titleEl.textContent = title;
  if (metaEl) metaEl.innerHTML = '';
  bodyEl.innerHTML = '<div class="hub-note">Loading…</div>';
  modal.style.display = 'flex';
  try {
    const r = await fetch(url);
    const d = await r.json().catch(() => ({}));
    if (!r.ok || d.error) {
      bodyEl.innerHTML = `<div class="hub-note error">${(d && d.error) || 'Failed to load model card.'}</div>`;
      return;
    }
    if (metaEl) metaEl.innerHTML = renderCardMeta(d);
    if (d.card) {
      renderMarkdownInto(bodyEl, d.card);
    } else {
      bodyEl.innerHTML = '<div class="hub-note">No model card (README) is bundled with this model.</div>';
    }
  } catch (e) {
    bodyEl.innerHTML = `<div class="hub-note error">Failed to load: ${e.message}</div>`;
  }
}
function renderCardMeta(d) {
  const rows = [];
  const esc = (s) => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const add = (k, v) => { if (v != null && v !== '') rows.push(`<div class="mck">${k}</div><div class="mcv">${v}</div>`); };
  add('Type', d.type ? esc(d.type.toUpperCase()) : null);
  add('Parameters', d.params ? esc(d.params) : null);
  add('Quantization', d.quantization ? esc(d.quantization) : null);
  add('Weights size', d.sizeBytes ? fmtBytes(d.sizeBytes) : null);
  if (d.type !== 'asr') add('Vision', d.supportsVision ? 'Yes' : 'No');
  if (d.imageSize && d.imageSize.width) add('Image size', `${d.imageSize.width}×${d.imageSize.height}`);
  const cfg = d.config || {};
  if (cfg.model_type) add('Architecture', esc(cfg.model_type));
  const ctx = cfg.max_seq_len || cfg.context_length || cfg.max_position_embeddings;
  if (ctx) add('Context length', esc(ctx));
  if (d.downloads != null) add('Downloads', Number(d.downloads).toLocaleString());
  if (d.likes != null) add('Likes', Number(d.likes).toLocaleString());
  if (Array.isArray(d.tags) && d.tags.length) {
    add('Tags', d.tags.slice(0, 10).map(t => `<span class="mtag">${esc(t)}</span>`).join(' '));
  }
  add('Source', d.source === 'huggingface' ? 'Hugging Face' : 'Installed');
  return rows.length ? `<div class="mcgrid">${rows.join('')}</div>` : '';
}
function initModelCardModal() {
  const modal = document.getElementById('modelCardModal');
  const close = document.getElementById('modelCardClose');
  if (close) close.addEventListener('click', () => { modal.style.display = 'none'; });
  if (modal) modal.addEventListener('click', (e) => { if (e.target === modal) modal.style.display = 'none'; });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && modal.style.display === 'flex') modal.style.display = 'none';
  });
}

async function loadModelAndActivate(name) {
  if (!name || _modelBusy) return;
  const select = document.getElementById('chatModelSelect');
  const option = select && Array.from(select.options).find(o => o.value === name);
  if (option && option.dataset.loaded === 'true') {
    if (typeof updateSelectedModelVisionState === 'function') updateSelectedModelVisionState();
    return; // already resident
  }
  // Incomplete weights can't load — surface it directly (a reset won't help).
  if (option && option.dataset.complete === 'false') {
    setModelStatus('Model weights are incomplete', 'error');
    showModelError(name, `${option.dataset.incompleteReason || 'The weights are incomplete'}. Re-download it from the Add Model tab.`);
    return;
  }

  _modelBusy = true;
  _pendingLoad = name;   // the list row shows "Loading…" for this model
  if (select) select.disabled = true;
  updateManageButtons();
  clearModelError();
  await resetLoadLog();
  // Loading a chat/VLM model evicts the currently-resident one — say so explicitly.
  const resident = select
    ? Array.from(select.options).filter(o => o.dataset.loaded === 'true' && o.value !== name).map(o => o.value)
    : [];
  const switchNote = resident.length ? `Unloading ${resident.join(', ')} — ` : '';
  setModelStatus(`${switchNote}Loading ${name}… preparing`, 'loading');
  setModelLoadBar('active');
  startLoadPolling(name);
  const hints = modelLoadHints(name);
  startLoadTicker(name, hints.est, hints.stages);
  try {
    const resp = await fetch('/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || 'load failed');
    const ev = Array.isArray(data.evicted) ? data.evicted : (data.evicted ? [data.evicted] : []);
    const evictedNote = ev.length ? ` · unloaded ${ev.join(', ')}` : '';
    const secs = (typeof data.load_seconds === 'number') ? data.load_seconds : null;
    const timeNote = (secs != null && secs > 0) ? ` in ${secs.toFixed(1)}s` : '';
    setModelStatus(`Ready: ${name}${timeNote}${evictedNote}`, 'ready');
    // A newly loaded model starts with a fresh context — clear the chat.
    newChat();
  } catch (err) {
    setModelStatus('Load failed — see details below', 'error');
    showModelError(name, err.message);
  } finally {
    stopLoadPolling();
    stopLoadTicker();
    if (select) select.disabled = false;
    _modelBusy = false;
    _pendingLoad = '';
    await pollLoadLogsOnce(name);   // flush any remaining load-log lines
    setModelLoadBar(null);
    await refreshCatalog();
    if (select) select.value = name;
    updateManageButtons();
    if (typeof updateSelectedModelVisionState === 'function') updateSelectedModelVisionState();
  }
}

// Live model-load feed. Log lines stream via Server-Sent Events so they appear
// as they happen (fast incremental polling is the fallback). Model load has no
// reliable percentage, so the bar is indeterminate and the status shows the ELF
// stage count when available — never a percentage.
let _loadPoll = null;
let _loadES = null;
let _logCursor = 0;
let _logLineCount = 0;

function applyLoadingStatus(ld, name) {
  if (!ld || ld.name !== name) return;
  _lastServerLoadUpdate = Date.now();
  const parts = [];
  // Percent first — it is what the eye goes to. `estimated` says whether it is
  // derived from counted stages or from elapsed-vs-expected time.
  if (ld.pct != null) parts.push(`${ld.pct}%`);

  if (ld.filesTotal && ld.filesDone != null) {
    // Real per-stage counter (older runtimes report one).
    parts.push(`stage ${ld.filesDone} of ${ld.filesTotal}`);
  } else if (ld.stagesTotal) {
    // Current runtimes load every stage in one bulk call and count nothing, so
    // show the scale of the work rather than a counter frozen at zero.
    parts.push(`${ld.stagesTotal} stages`);
  }
  parts.push(fmtDuration(ld.elapsedS));

  // The countdown. Say "~" while it is an estimate so the number is not read as
  // a promise, and stop counting down past zero on a slower-than-expected load.
  const remain = (ld.remainingS != null)
    ? ld.remainingS
    : (ld.etaS != null ? Math.max(0, ld.etaS - ld.elapsedS) : null);
  if (remain != null) {
    parts.push(remain > 0 ? `~${fmtDuration(remain)} left` : 'finishing…');
  }

  setModelStatus(`Loading ${name} · ${parts.join(' · ')}`, 'loading');
  setModelLoadBar(ld.pct != null ? ld.pct : 'active');
}

function processLoadUpdate(d, name) {
  if (!d) return;
  if (Array.isArray(d.lines)) appendLoadLogLines(d.lines);
  if (d.loading) applyLoadingStatus(d.loading, name);
}

// Single poll — the SSE fallback, and the flush of the final lines once done.
async function pollLoadLogsOnce(name) {
  let d = null;
  try {
    const r = await fetch('/models/logs?after=' + _logCursor);
    d = await r.json();
  } catch (e) { return null; }
  processLoadUpdate(d, name);
  return d;
}

function startLoadPolling(name) {
  stopLoadPolling();
  // Stream log lines as they happen; fall back to fast polling if SSE fails.
  if (typeof EventSource !== 'undefined') {
    try {
      const es = new EventSource('/models/logs/stream?after=' + _logCursor);
      _loadES = es;
      let gotData = false;
      es.onmessage = (e) => {
        gotData = true;
        let d = null; try { d = JSON.parse(e.data); } catch (_) { return; }
        processLoadUpdate(d, name);
      };
      es.addEventListener('done', (e) => {
        let err = false;
        try { err = !!(JSON.parse(e.data || '{}').error); } catch (_) { /* ignore */ }
        stopLoadPolling();
        // A sustained control-API outage ends the stream with an error while the
        // load is still running — keep progress alive via the polling fallback.
        if (err && _modelBusy) startPollFallback(name);
      });
      es.onerror = () => {
        // The server fires 'done' before closing a completed stream, so reaching
        // onerror without any data means it never connected → fall back to poll.
        if (!gotData) { stopLoadPolling(); startPollFallback(name); }
      };
      return;
    } catch (e) { /* fall through to polling */ }
  }
  startPollFallback(name);
}

function startPollFallback(name) {
  if (_loadPoll) clearInterval(_loadPoll);
  _loadPoll = setInterval(() => { pollLoadLogsOnce(name); }, 400);
}

function stopLoadPolling() {
  if (_loadPoll) { clearInterval(_loadPoll); _loadPoll = null; }
  if (_loadES) { try { _loadES.close(); } catch (e) { /* ignore */ } _loadES = null; }
}

// Reset the load-log view and advance the cursor past everything already logged
// so only THIS load's lines are shown (the server's line sequence is monotonic).
async function resetLoadLog() {
  _logLineCount = 0;
  const view = document.getElementById('modelLogView');
  if (view) view.textContent = '';
  updateLoadLogCount(0);
  try {
    const r = await fetch('/models/logs?after=999999999');
    const d = await r.json();
    _logCursor = d.seq || 0;
    if (d.available === false && view) view.textContent = 'Live load logs are unavailable (stdout tap disabled).';
  } catch (e) { /* keep the prior cursor so earlier loads' lines stay hidden */ }
}

function updateLoadLogCount(n) {
  const el = document.getElementById('modelLogCount');
  if (el) el.textContent = n ? `${n} line${n === 1 ? '' : 's'}` : '';
}

function appendLoadLogLines(lines) {
  const view = document.getElementById('modelLogView');
  if (!Array.isArray(lines) || !lines.length) return;
  const mirror = _loadMirror.active ? _loadMirror.logEl : null;
  if (!view && !mirror) return;
  const atBottom = view ? (view.scrollTop + view.clientHeight >= view.scrollHeight - 8) : false;
  const mAtBottom = mirror ? (mirror.scrollTop + mirror.clientHeight >= mirror.scrollHeight - 8) : false;
  const frag = document.createDocumentFragment();
  const mfrag = mirror ? document.createDocumentFragment() : null;
  let added = 0;
  lines.forEach(l => {
    const seq = (l && l.seq) || 0;
    // Skip lines already shown — an SSE reconnect/replay re-sends earlier lines.
    if (seq && seq <= _logCursor) return;
    const t = (l && l.text) || '';
    const cls = /^Done loading/.test(t) ? 'model-log-line-done'
      : /ok=0|fail|error/i.test(t) ? 'model-log-line-err' : '';
    const div = document.createElement('div');
    div.textContent = t;
    if (cls) div.className = cls;
    frag.appendChild(div);
    if (mfrag) {
      const m = document.createElement('div');
      m.textContent = t;
      if (cls) m.className = cls;
      mfrag.appendChild(m);
    }
    _logLineCount++;
    if (seq > _logCursor) _logCursor = seq;
    added++;
  });
  if (!added) return;
  if (view) {
    view.appendChild(frag);
    if (atBottom) view.scrollTop = view.scrollHeight;
  }
  if (mirror) {
    mirror.appendChild(mfrag);
    if (mAtBottom) mirror.scrollTop = mirror.scrollHeight;
  }
  updateLoadLogCount(_logLineCount);
}

// ---- Prominent model-load failure -------------------------------------
function showModelError(name, message) {
  const box = document.getElementById('modelLoadError');
  const msg = document.getElementById('modelLoadErrorMsg');
  const title = box && box.querySelector('.model-load-error-title');
  if (title) title.textContent = name ? `“${name}” failed to load` : 'Model failed to load';
  if (msg) msg.textContent = message || 'Unknown error';
  if (box) { box.style.display = 'flex'; box.dataset.model = name || ''; }
  const det = document.getElementById('modelLogDetails');   // reveal what happened
  if (det) det.open = true;
}
function clearModelError() {
  const box = document.getElementById('modelLoadError');
  if (box) { box.style.display = 'none'; box.dataset.model = ''; }
}

// Model-load progress bar. Model load has no reliable percentage, so this is an
// INDETERMINATE animated bar: setModelLoadBar('active') shows it, null hides it.
function setModelLoadBar(state) {
  const bar = document.getElementById('modelLoadBar');
  const panel = document.getElementById('modelLoadProgress');
  if (!bar) return;
  const fill = bar.querySelector('.model-load-fill');
  if (!state) {
    bar.style.display = 'none';
    if (fill) { fill.style.width = '0%'; fill.classList.remove('indeterminate'); }
    if (panel) panel.classList.remove('is-busy');
    return;
  }
  bar.style.display = 'block';
  if (fill) {
    if (typeof state === 'number' && isFinite(state)) {
      // A real percentage: fill to it and drop the barber-pole animation.
      fill.classList.remove('indeterminate');
      fill.style.width = `${Math.max(0, Math.min(100, state))}%`;
    } else {
      // No percentage available — keep the indeterminate sweep.
      fill.style.width = '100%';
      fill.classList.add('indeterminate');
    }
  }
  // Pin the status panel to the top and bring it into view — with per-row Load
  // buttons the click may happen far below it (especially on a small screen).
  if (panel) {
    panel.classList.add('is-busy');
    if (panel.scrollIntoView) {
      try { panel.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } catch (e) { /* ignore */ }
    }
  }
}

// ---- Hugging Face download --------------------------------------------

let _hubEnabled = false;

async function initHubControls() {
  // Models tab: search + type/size/family/sort filters + rescan act on the
  // DOWNLOADED list only.
  const modelSearch = document.getElementById('modelSearchInput');
  if (modelSearch) modelSearch.addEventListener('input', updateManageButtons);
  ['modelFilterType', 'modelFilterParams', 'modelFilterFamily', 'modelSortBy'].forEach(id => {
    const el = document.getElementById(id); if (el) el.addEventListener('change', renderInstalledList);
  });
  updateAsrModelIndicator();
  const modelRefresh = document.getElementById('modelRefreshButton');
  if (modelRefresh) modelRefresh.addEventListener('click', () => refreshCatalog());

  // Add Model tab: its own search + filters + refresh act on the Hugging Face list.
  const hubSearch = document.getElementById('hubSearchInput');
  if (hubSearch) hubSearch.addEventListener('input', applyHubFilters);
  const hubRefresh = document.getElementById('hubRefreshButton');
  if (hubRefresh) hubRefresh.addEventListener('click', () => { _hubLoaded = true; loadHubModels(); });
  ['hubFilterType', 'hubFilterParams', 'hubFilterFamily', 'hubSortBy'].forEach(id => {
    const el = document.getElementById(id); if (el) el.addEventListener('change', applyHubFilters);
  });

  // Probe availability (online + allowed). The Add Model tab is always
  // visible; when the hub is offline/disabled its panel shows a note instead.
  let enabled = false;
  let data = null;
  try {
    const resp = await fetch('/models/status');
    data = await resp.json();
    enabled = !!(data && data.hubEnabled);
  } catch (err) { enabled = false; }
  _hubEnabled = enabled;
  renderDiskInfo(data && data.disk);   // NVMe free space (from the same status call)
  renderInstalledList();   // refresh the empty-state hint now that hub status is known
}

// Every compatible model is listed up front; the search box + dropdowns filter it.
let _hubAllModels = [];
let _hubLoaded = false;

function ensureHubLoaded() {
  if (_hubLoaded) return;
  _hubLoaded = true;
  loadHubModels();
}

async function loadHubModels() {
  const results = document.getElementById('modelHubList');
  if (results) results.innerHTML = '<div class="hub-note">Loading available models…</div>';
  try {
    const resp = await fetch('/models/hub/search?q=');   // empty query = list all
    const data = await resp.json();
    if (!data || !data.enabled) {
      _hubAllModels = [];
      if (results) results.innerHTML = `<div class="hub-note">${(data && data.error) || 'Hugging Face is offline or disabled.'}</div>`;
      return;
    }
    _hubAllModels = (data.results || []).map(m => {
      const b = hubModelParams(m.repoId);
      // The server classifies from Hub metadata (pipeline_tag/tags), which is
      // reliable; the repo-name guess is only for older servers.
      const t = m.type ? m.type.toUpperCase() : hubModelType(m.repoId);
      return { ...m, _type: t, _params: b, _bucket: hubParamsBucket(b), _family: hubModelFamily(m.repoId) };
    });
    populateHubFamilyFilter();
    applyHubFilters();
  } catch (err) {
    if (results) results.innerHTML = `<div class="hub-note error">Could not load models: ${err.message}</div>`;
  }
}

// Best-effort LLM vs VLM from the repo name (a true classification needs the
// model config, which we don't have at list time).
const HUB_VLM_HINTS = ['vlm', '-vl', 'vl-', '_vl', 'vl2', 'vl3', 'vl4', 'vision', 'multimodal',
  'omni', 'internvl', 'llava', 'pixtral', 'molmo', 'minicpm-v', 'cpm-v', 'idefics', 'paligemma', 'siglip'];
function hubModelType(repoId) {
  const n = (repoId.split('/').pop() || repoId).toLowerCase();
  if (n.includes('whisper')) return 'ASR';
  return HUB_VLM_HINTS.some(k => n.includes(k)) ? 'VLM' : 'LLM';
}
function hubModelParams(repoId) {
  const name = (repoId.split('/').pop() || repoId);
  let m = name.match(/(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])/);
  if (m) return parseFloat(m[1]);
  m = name.match(/(\d+(?:\.\d+)?)\s*[mM](?![a-zA-Z])/);
  if (m) return parseFloat(m[1]) / 1000;
  return null;
}
function hubParamsBucket(b) {
  if (b == null) return null;
  if (b < 1) return '0-1';
  if (b < 4) return '1-4';
  if (b < 8) return '4-8';
  return '8-999';
}
const HUB_FAMILIES = ['Qwen', 'Llama', 'Gemma', 'Mistral', 'Mixtral', 'Phi', 'DeepSeek', 'Falcon', 'Yi', 'Vicuna', 'StableLM', 'SmolLM', 'TinyLlama', 'Granite', 'InternLM', 'InternVL', 'Baichuan', 'ChatGLM', 'OLMo', 'Command', 'Molmo', 'Pixtral', 'LLaVA', 'MiniCPM'];
function hubModelFamily(repoId) {
  const name = (repoId.split('/').pop() || repoId);
  for (const f of HUB_FAMILIES) { if (new RegExp(f, 'i').test(name)) return f; }
  const m = name.match(/^[A-Za-z]+/);
  return m ? (m[0][0].toUpperCase() + m[0].slice(1)) : 'Other';
}

function populateHubFamilyFilter() {
  const sel = document.getElementById('hubFilterFamily');
  if (!sel) return;
  const cur = sel.value;
  const fams = Array.from(new Set(_hubAllModels.map(m => m._family))).sort((a, b) => a.localeCompare(b));
  sel.innerHTML = '<option value="">All families</option>' + fams.map(f => `<option value="${f}">${f}</option>`).join('');
  if (fams.includes(cur)) sel.value = cur;
}

function applyHubFilters() {
  const text = (document.getElementById('hubSearchInput')?.value || '').trim().toLowerCase();
  const ft = document.getElementById('hubFilterType')?.value || '';
  const fp = document.getElementById('hubFilterParams')?.value || '';
  const ff = document.getElementById('hubFilterFamily')?.value || '';
  const filtered = _hubAllModels.filter(m => {
    // Fully-installed models live in the Installed section above; only offer the
    // Hugging Face row for new models and incomplete ones (which need re-download).
    if (m.alreadyInCatalog && m.catalogComplete !== false) return false;
    if (text && !m.repoId.toLowerCase().includes(text)) return false;
    if (ft && m._type !== ft) return false;
    if (fp && m._bucket !== fp) return false;
    if (ff && m._family !== ff) return false;
    return true;
  });
  const sort = document.getElementById('hubSortBy')?.value || 'downloads';
  filtered.sort((a, b) => {
    switch (sort) {
      case 'size-desc': return (b.sizeBytes || 0) - (a.sizeBytes || 0);
      case 'size-asc': return (a.sizeBytes || Infinity) - (b.sizeBytes || Infinity);
      case 'name': return a.repoId.localeCompare(b.repoId);
      case 'params-desc': return (b._params || 0) - (a._params || 0);
      case 'params-asc': return (a._params == null ? Infinity : a._params) - (b._params == null ? Infinity : b._params);
      default: return (b.downloads || 0) - (a.downloads || 0);
    }
  });
  renderHubResults({ enabled: true, results: filtered });
  const count = document.getElementById('hubResultCount');
  if (count) count.textContent = filtered.length ? `${filtered.length} to download` : '';
}

function renderHubResults(data) {
  const results = document.getElementById('modelHubList');
  if (!results) return;
  results.innerHTML = '';
  const items = (data && data.results) || [];
  if (!data.enabled) {
    results.innerHTML = '<div class="hub-note">Hugging Face is offline or disabled.</div>';
    return;
  }
  if (items.length === 0) {
    const searching = !!(document.getElementById('hubSearchInput')?.value || '').trim();
    results.innerHTML = searching
      ? '<div class="hub-note">No Hugging Face models match your search.</div>'
      : '<div class="hub-note">Everything available has already been downloaded — see the Models tab.</div>';
    return;
  }
  items.forEach(item => {
    const row = document.createElement('div');
    row.className = 'hub-result';
    const meta = document.createElement('div');
    meta.className = 'hub-result-meta';
    const parts = [];
    if (item.downloads != null) parts.push(`${item.downloads.toLocaleString()} downloads`);
    const sub = parts.length ? parts.join(' · ') : '';
    const t = item._type || hubModelType(item.repoId);
    const p = (item._params != null) ? item._params : hubModelParams(item.repoId);
    const fam = item._family || hubModelFamily(item.repoId);
    const incomplete = item.alreadyInCatalog && item.catalogComplete === false;
    // Download size (≈ on-disk footprint) is called out as its own badge.
    const sizeBadge = item.sizeBytes
      ? `<span class="hub-badge hub-badge-size" title="Download size — space it will take on the NVMe">⬇ ${fmtBytes(item.sizeBytes)}</span>`
      : '<span class="hub-badge hub-badge-size hub-badge-muted" title="Download size unknown (open ℹ for details)">⬇ size —</span>';
    const badges = `<span class="hub-badge hub-badge-${t.toLowerCase()}">${t}</span>` +
      (p != null ? `<span class="hub-badge">${p}B</span>` : '') +
      sizeBadge +
      `<span class="hub-badge hub-badge-fam">${fam}</span>` +
      (incomplete ? `<span class="hub-badge hub-badge-warn" title="Local copy is missing files — re-download to fix">⚠ incomplete</span>` : '');
    meta.innerHTML = `<span class="hub-repo">${item.repoId}</span><span class="hub-badges">${badges}</span><span class="hub-sub">${sub}</span>`;
    const info = document.createElement('button');
    info.className = 'hub-info';
    info.type = 'button';
    info.title = `Model card & metadata for ${item.repoId}`;
    info.textContent = 'ℹ';
    info.addEventListener('click', () => showHubCard(item.repoId));
    const btn = document.createElement('button');
    btn.className = 'setting-button hub-download-btn' + (incomplete ? ' hub-redownload' : '');
    btn.textContent = incomplete ? 'Re-download' : (item.alreadyInCatalog ? 'In catalog' : 'Download');
    btn.disabled = !!(item.alreadyInCatalog && !incomplete);
    if (incomplete) btn.title = 'The on-disk copy is incomplete — download again to repair it';
    btn.addEventListener('click', () => hubDownload(item.repoId, row, btn));
    row.appendChild(meta);
    row.appendChild(info);
    row.appendChild(btn);
    results.appendChild(row);
  });
}

async function hubDownload(repoId, row, btn) {
  if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }
  let bar = row.querySelector('.hub-progress');
  if (!bar) {
    bar = document.createElement('div');
    bar.className = 'hub-progress';
    bar.innerHTML = '<div class="hub-progress-fill"></div><span class="hub-progress-label"></span>';
    row.appendChild(bar);
  }
  let progMeta = row.querySelector('.hub-progress-meta');
  if (!progMeta) {
    progMeta = document.createElement('div');
    progMeta.className = 'hub-progress-meta';
    row.appendChild(progMeta);
  }
  const fill = bar.querySelector('.hub-progress-fill');
  const label = bar.querySelector('.hub-progress-label');

  try {
    const resp = await fetch('/models/hub/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoId })
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let evt;
        try { evt = JSON.parse(line); } catch (e) { continue; }
        if (evt.state === 'downloading') {
          const pct = evt.pct != null ? evt.pct : null;
          if (fill) fill.style.width = (pct != null ? pct : 5) + '%';
          if (label) {
            const sizeStr = evt.total
              ? `${fmtBytes(evt.downloaded)} / ${fmtBytes(evt.total)}`
              : (evt.downloaded ? fmtBytes(evt.downloaded) : '');
            const pctStr = pct != null ? `${pct}%` : 'Downloading…';
            label.textContent = sizeStr ? `${pctStr} · ${sizeStr}` : pctStr;
          }
          if (progMeta) {
            const bits = [];
            if (evt.speedBps) bits.push(`${fmtBytes(evt.speedBps)}/s`);
            if (evt.etaS != null) bits.push(`~${fmtDuration(evt.etaS)} left`);
            progMeta.textContent = bits.join(' · ');
          }
          if (btn) btn.textContent = 'Downloading…';
        } else if (evt.state === 'resolving') {
          if (label) label.textContent = 'Resolving…';
          if (progMeta) progMeta.textContent = '';
        } else if (evt.state === 'done') {
          if (fill) fill.style.width = '100%';
          if (label) label.textContent = evt.total ? `Done · ${fmtBytes(evt.total)}` : 'Done';
          if (progMeta) progMeta.textContent = '';
          if (btn) btn.textContent = 'In catalog';
          if (btn) btn.disabled = true;
          await refreshCatalog();   // the model now appears in the Installed section.
          refreshDiskInfo();        // free space just dropped
          // Mark it installed so the NEXT re-filter drops it from the Hugging Face
          // section — but don't re-render now: a re-render here would wipe any OTHER
          // download still in flight (its row + progress live in the same list).
          const hit = _hubAllModels.find(x => x.repoId === repoId);
          if (hit) { hit.alreadyInCatalog = true; hit.catalogComplete = true; }
        } else if (evt.state === 'error') {
          if (label) label.textContent = 'Error: ' + (evt.message || 'failed');
          bar.classList.add('error');
          if (btn) { btn.disabled = false; btn.textContent = 'Retry'; }
        }
      }
    }
  } catch (err) {
    if (label) label.textContent = 'Error: ' + err.message;
    if (btn) { btn.disabled = false; btn.textContent = 'Retry'; }
  }
}

// ---- Font customization -----------------------------------------------

const FONT_PRESETS = {
  'Inter': "'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
  'System Sans': "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
  'Georgia': "Georgia, 'Times New Roman', serif",
  'JetBrains Mono': "'JetBrains Mono', ui-monospace, 'Courier New', monospace",
  'Courier': "'Courier New', Courier, monospace"
};

function applyFont(family, size) {
  const root = document.documentElement;
  const stack = FONT_PRESETS[family] || (family ? `'${family}', system-ui, sans-serif` : FONT_PRESETS['Inter']);
  root.style.setProperty('--ui-font', stack);
  if (size) root.style.setProperty('--ui-font-size', size + 'px');
}

// ---- Accent colour + SiMa multicolor accents ----------------------------
const ACCENT_PRESETS = [
  { name: 'Teal (default)', hex: '' },
  { name: 'Sky', hex: '#3a86ec' },
  { name: 'Indigo', hex: '#6366f1' },
  { name: 'Violet', hex: '#a855f7' },
  { name: 'Emerald', hex: '#1fb866' },
  { name: 'Lime', hex: '#7bb318' },
  { name: 'Amber', hex: '#f2801d' },
  { name: 'Rose', hex: '#ef4d6a' },
];

function _accentLuminance(hex) {
  const m = /^#?([0-9a-fA-F]{6})$/.exec(hex || '');
  if (!m) return 0.5;
  const n = parseInt(m[1], 16), r = (n >> 16 & 255) / 255, g = (n >> 8 & 255) / 255, b = (n & 255) / 255;
  const f = (c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}

// Set (or clear, when hex is falsy) the accent CSS vars, deriving the shades so a
// single colour recolours buttons, links, focus rings and highlights.
function applyAccent(hex) {
  const r = document.documentElement.style;
  const vars = ['--accent', '--accent-2', '--accent-strong', '--accent-weak', '--accent-contrast', '--user-bubble'];
  if (!hex) { vars.forEach(v => r.removeProperty(v)); return; }
  r.setProperty('--accent', hex);
  r.setProperty('--accent-2', `color-mix(in srgb, ${hex} 85%, #fff)`);
  r.setProperty('--accent-strong', `color-mix(in srgb, ${hex} 80%, #000)`);
  r.setProperty('--accent-weak', `color-mix(in srgb, ${hex} 15%, transparent)`);
  r.setProperty('--accent-contrast', _accentLuminance(hex) > 0.6 ? '#0b1015' : '#ffffff');
  r.setProperty('--user-bubble', `color-mix(in srgb, ${hex} 88%, #000)`);
}

function setSpectrum(on) {
  document.body.classList.toggle('spectrum-accents', on);
  const cb = document.getElementById('toggleSpectrum'); if (cb) cb.checked = on;
  try { localStorage.setItem('studioSpectrum', on ? '1' : '0'); } catch (e) { /* ignore */ }
}

function markAccentSwatch(hex) {
  document.querySelectorAll('#accentSwatches .accent-swatch').forEach(b => {
    b.setAttribute('aria-pressed', (!b.dataset.spectrum && (b.dataset.hex || '') === (hex || '')) ? 'true' : 'false');
  });
}

function selectAccent(hex) {
  applyAccent(hex);
  try { localStorage.setItem('studioAccent', hex || ''); } catch (e) { /* ignore */ }
  const custom = document.getElementById('accentCustom');
  if (custom && hex) custom.value = hex;
  markAccentSwatch(hex);
}

function initAccentControls() {
  let saved = '';
  try { saved = localStorage.getItem('studioAccent') || ''; } catch (e) { /* ignore */ }
  applyAccent(saved);
  let specOn = true;   // multicolor accents on by default (matches the marketing look)
  try { specOn = localStorage.getItem('studioSpectrum') !== '0'; } catch (e) { /* ignore */ }
  document.body.classList.toggle('spectrum-accents', specOn);

  const wrap = document.getElementById('accentSwatches');
  if (wrap) {
    wrap.innerHTML = '';
    ACCENT_PRESETS.forEach(p => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'accent-swatch';
      b.style.background = p.hex || '#12b3a2';
      b.dataset.hex = p.hex; b.title = p.name; b.setAttribute('aria-label', p.name);
      b.addEventListener('click', () => selectAccent(p.hex));
      wrap.appendChild(b);
    });
    const sp = document.createElement('button');
    sp.type = 'button'; sp.className = 'accent-swatch spectrum';
    sp.dataset.spectrum = '1'; sp.title = 'SiMa multicolor'; sp.setAttribute('aria-label', 'SiMa multicolor accents');
    sp.addEventListener('click', () => { selectAccent(''); setSpectrum(true); });
    wrap.appendChild(sp);
    markAccentSwatch(saved);
  }
  const custom = document.getElementById('accentCustom');
  if (custom) {
    if (saved) custom.value = saved;
    custom.addEventListener('input', () => selectAccent(custom.value));
  }
  const spectrum = document.getElementById('toggleSpectrum');
  if (spectrum) {
    spectrum.checked = specOn;
    spectrum.addEventListener('change', () => setSpectrum(spectrum.checked));
  }
}

function getSavedFont() {
  const cfg = window.SIMA_CONFIG || {};
  return {
    family: localStorage.getItem('studioFontFamily') || cfg.defaultFontFamily || 'Inter',
    size: parseInt(localStorage.getItem('studioFontSize') || cfg.defaultFontSize || 15, 10),
    custom: localStorage.getItem('studioFontCustom') || ''
  };
}

function applySavedFont() {
  const saved = getSavedFont();
  const family = saved.custom || saved.family;
  applyFont(family, saved.size);
}

function initFontControls() {
  const select = document.getElementById('fontFamilySelect');
  const sizeRange = document.getElementById('fontSizeRange');
  const sizeValue = document.getElementById('fontSizeValue');
  const custom = document.getElementById('fontCustomInput');
  const saved = getSavedFont();

  if (select) {
    while (select.firstChild) select.removeChild(select.firstChild);
    Object.keys(FONT_PRESETS).forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      select.appendChild(opt);
    });
    select.value = FONT_PRESETS[saved.family] ? saved.family : 'Inter';
    select.addEventListener('change', () => {
      if (custom) custom.value = '';
      localStorage.removeItem('studioFontCustom');
      localStorage.setItem('studioFontFamily', select.value);
      applyFont(select.value, currentFontSize());
    });
  }
  if (sizeRange) {
    sizeRange.value = saved.size;
    if (sizeValue) sizeValue.textContent = saved.size + 'px';
    sizeRange.addEventListener('input', () => {
      if (sizeValue) sizeValue.textContent = sizeRange.value + 'px';
      localStorage.setItem('studioFontSize', sizeRange.value);
      applyFont(custom && custom.value ? custom.value : (select ? select.value : 'Inter'), sizeRange.value);
    });
  }
  if (custom) {
    custom.value = saved.custom || '';
    const applyCustom = () => {
      const fam = custom.value.trim();
      if (fam) {
        localStorage.setItem('studioFontCustom', fam);
        applyFont(fam, currentFontSize());
      } else {
        localStorage.removeItem('studioFontCustom');
        applyFont(select ? select.value : 'Inter', currentFontSize());
      }
    };
    custom.addEventListener('change', applyCustom);
    custom.addEventListener('keydown', (e) => { if (e.key === 'Enter') applyCustom(); });
  }
  applySavedFont();
}

function currentFontSize() {
  const sizeRange = document.getElementById('fontSizeRange');
  return sizeRange ? sizeRange.value : (getSavedFont().size);
}

// ---- Text-to-speech (PiperTTS) toggle ---------------------------------

function isTtsEnabled() {
  const el = document.getElementById('toggleTTS');
  return el ? el.checked : true;
}

function initTtsToggle() {
  const el = document.getElementById('toggleTTS');
  if (!el) return;
  const saved = localStorage.getItem('studioTtsEnabled');
  if (saved !== null) el.checked = saved === 'true';
  el.addEventListener('change', () => {
    localStorage.setItem('studioTtsEnabled', el.checked ? 'true' : 'false');
    // Turning it off mid-reply: stop any audio already playing/queued.
    if (!el.checked && typeof stopAudio === 'function') {
      try { stopAudio(); } catch (e) { /* ignore */ }
    }
  });
}

// ---- Generation settings (max response tokens) ------------------------

function getMaxTokens() {
  const el = document.getElementById('maxTokensRange');
  return el ? parseInt(el.value, 10) : 512;
}

// Whether reasoning models should think. On by default; when off we ask the
// server to disable thinking (/no_think). No effect on non-reasoning models.
function getThinkingEnabled() {
  const el = document.getElementById('toggleThinking');
  if (el) return el.checked;
  try { return localStorage.getItem('studioThinking') !== '0'; } catch (e) { return true; }
}

function initGenerationControls() {
  const range = document.getElementById('maxTokensRange');
  const value = document.getElementById('maxTokensValue');
  const think = document.getElementById('toggleThinking');
  if (think) {
    let on = true;
    try { on = localStorage.getItem('studioThinking') !== '0'; } catch (e) { /* ignore */ }
    think.checked = on;
    think.addEventListener('change', () => {
      try { localStorage.setItem('studioThinking', think.checked ? '1' : '0'); } catch (e) { /* ignore */ }
    });
  }
  if (!range) return;
  const cfgDefault = parseInt(window.SIMA_CONFIG?.defaultMaxTokens, 10);
  const saved = parseInt(localStorage.getItem('studioMaxTokens'), 10);
  if (Number.isFinite(saved)) range.value = saved;
  else if (Number.isFinite(cfgDefault)) range.value = cfgDefault;
  const sync = () => { if (value) value.textContent = range.value; };
  sync();
  range.addEventListener('input', () => {
    sync();
    localStorage.setItem('studioMaxTokens', range.value);
  });
}

// ---- Full-screen toggle -----------------------------------------------

function toggleFullscreen() {
  const el = document.documentElement;
  const isFs = document.fullscreenElement || document.webkitFullscreenElement;
  let p;
  if (isFs) {
    p = (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  } else {
    p = (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
  }
  if (p && typeof p.catch === 'function') p.catch(() => {});
}

function initFullscreenButton() {
  const btn = document.getElementById('fullscreenButton');
  if (!btn) return;
  const supported = document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen;
  if (!supported) { btn.style.display = 'none'; return; }
  btn.addEventListener('click', () => { try { toggleFullscreen(); } catch (e) { /* ignore */ } });
  const sync = () => {
    const on = !!(document.fullscreenElement || document.webkitFullscreenElement);
    btn.classList.toggle('is-fullscreen', on);
    btn.title = on ? 'Exit full screen' : 'Full screen';
  };
  document.addEventListener('fullscreenchange', sync);
  document.addEventListener('webkitfullscreenchange', sync);
  sync();
}

// ---- Collapsible camera dock ------------------------------------------

function revealCameraForVisionModel() {
  const section = document.getElementById('cameraSection');
  const btn = document.getElementById('cameraCollapseBtn');
  if (!section) return;
  section.classList.remove('collapsed');
  if (btn) {
    btn.title = 'Collapse camera';
    btn.setAttribute('aria-label', btn.title);
  }
  try { localStorage.setItem('studioCameraCollapsed', '0'); } catch (e) { /* ignore */ }
  window.dispatchEvent(new Event('resize'));
}

function initCameraCollapse() {
  const section = document.getElementById('cameraSection');
  const btn = document.getElementById('cameraCollapseBtn');
  if (!section || !btn) return;
  const KEY = 'studioCameraCollapsed';
  const apply = (collapsed) => {
    section.classList.toggle('collapsed', collapsed);
    btn.title = collapsed ? 'Expand camera' : 'Collapse camera';
    btn.setAttribute('aria-label', btn.title);
  };
  // Collapsed by default (chat-first); respect the user's saved preference once set.
  let collapsed = true;
  try {
    const saved = localStorage.getItem(KEY);
    if (saved !== null) collapsed = saved === '1';
  } catch (e) { /* ignore */ }
  apply(collapsed);
  btn.addEventListener('click', () => {
    collapsed = !section.classList.contains('collapsed');
    apply(collapsed);
    try { localStorage.setItem(KEY, collapsed ? '1' : '0'); } catch (e) { /* ignore */ }
    // Re-fit the live feed when expanding (the dock size just changed).
    if (!collapsed && cameraPreview && cameraPreview.videoWidth > 0) {
      window.dispatchEvent(new Event('resize'));
    }
  });
}

// ---- Full-screen Vision (immersive camera / image + VLM Q&A) ----------

let _visionSource = 'camera';        // 'camera' | 'image'
let _visionMirrorObserver = null;    // mirrors the latest assistant answer
let _visionRecording = false;        // true while the vision mic is capturing voice
const VISION_ASK_HINT = 'Tap the mic and ask about what you see.';

function initVision() {
  const modal = document.getElementById('visionModal');
  if (!modal) return;
  const openBtn = document.getElementById('cameraVisionBtn');
  const closeBtn = document.getElementById('visionCloseBtn');
  const camBtn = document.getElementById('visionSourceCamera');
  const imgBtn = document.getElementById('visionSourceImage');
  const uploadBtn = document.getElementById('visionUploadBtn');
  const askBtn = document.getElementById('visionAskBtn');
  const micBtn = document.getElementById('visionMicBtn');
  const input = document.getElementById('visionInput');
  const chips = document.getElementById('visionChips');

  if (openBtn) openBtn.addEventListener('click', openVision);
  if (closeBtn) closeBtn.addEventListener('click', closeVision);
  if (camBtn) camBtn.addEventListener('click', () => setVisionSource('camera'));
  if (imgBtn) imgBtn.addEventListener('click', () => setVisionSource('image'));
  if (uploadBtn) uploadBtn.addEventListener('click', pickVisionImage);
  const boardBtn = document.getElementById('visionBoardBtn');
  if (boardBtn) boardBtn.addEventListener('click', snapVisionBoardCamera);
  if (micBtn) micBtn.addEventListener('click', toggleVisionMic);
  const loopBtn = document.getElementById('visionLoopBtn');
  if (loopBtn) loopBtn.addEventListener('click', toggleVisionLoop);
  if (askBtn) askBtn.addEventListener('click', () => askVision(input ? input.value : ''));
  if (input) input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); askVision(input.value); }
  });
  if (chips) chips.addEventListener('click', (e) => {
    const chip = e.target.closest('.vision-chip');
    if (chip) askVision(chip.textContent.trim());
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeVision();
  });
}

function openVision() {
  const modal = document.getElementById('visionModal');
  if (!modal) return;
  modal.style.display = 'flex';
  document.body.classList.add('vision-open');
  // If the model is vision-capable but the camera never started, start it now.
  if (!mediaStream && !isLlmOnlyMode()) {
    try { startCamera().then(() => setVisionSource(_visionSource)).catch(() => {}); } catch (e) { /* ignore */ }
  }
  setVisionSource('camera');
  refreshVisionModelState();
  setVisionQuestion('');
  setVisionAskHint('');
  startVisionMirror();
  // Best-effort browser fullscreen for true immersion.
  try {
    const rf = modal.requestFullscreen || modal.webkitRequestFullscreen;
    if (rf) { const p = rf.call(modal); if (p && p.catch) p.catch(() => {}); }
  } catch (e) { /* ignore */ }
  const input = document.getElementById('visionInput');
  if (input) setTimeout(() => input.focus(), 60);
}

function closeVision() {
  const modal = document.getElementById('visionModal');
  if (!modal) return;
  stopVisionLoop();          // end any continuous camera loop
  modal.style.display = 'none';
  document.body.classList.remove('vision-open');
  cancelVisionRecording();   // stop any in-progress voice capture + release the mic
  stopVisionMirror();
  // Release this consumer of the shared stream; the rail keeps the camera on.
  const v = document.getElementById('visionVideo');
  if (v) v.srcObject = null;
  try {
    if (document.fullscreenElement || document.webkitFullscreenElement) {
      const p = (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      if (p && p.catch) p.catch(() => {});
    }
  } catch (e) { /* ignore */ }
}

function setVisionSource(source) {
  _visionSource = source;
  const camBtn = document.getElementById('visionSourceCamera');
  const imgBtn = document.getElementById('visionSourceImage');
  const video = document.getElementById('visionVideo');
  const image = document.getElementById('visionImage');
  if (camBtn) camBtn.classList.toggle('is-active', source === 'camera');
  if (imgBtn) imgBtn.classList.toggle('is-active', source === 'image');
  if (source === 'camera') {
    if (image) image.style.display = 'none';
    if (video) {
      video.style.display = '';
      if (mediaStream && video.srcObject !== mediaStream) video.srcObject = mediaStream;
    }
    setVisionHint(mediaStream ? '' : 'The camera is off — enable it, or switch to Image.');
  } else {
    if (video) video.style.display = 'none';
    if (image) image.style.display = image.src ? '' : 'none';
    setVisionHint(image && image.src ? '' : 'No image loaded — press “Upload Image”.');
  }
  refreshVisionModelState();
}

function setVisionHint(text) {
  const hint = document.getElementById('visionStageHint');
  if (!hint) return;
  hint.textContent = text || '';
  hint.style.display = text ? 'block' : 'none';
}

function refreshVisionModelState() {
  const pill = document.getElementById('visionModelPill');
  const askBtn = document.getElementById('visionAskBtn');
  const micBtn = document.getElementById('visionMicBtn');
  const model = getSelectedChatModel();
  const isVlm = selectedChatModelSupportsVision();
  if (pill) pill.textContent = model ? `${model} · ${isVlm ? 'vision' : 'text-only'}` : 'No model loaded';
  if (askBtn) askBtn.disabled = !isVlm;
  if (micBtn) micBtn.disabled = !isVlm;
  if (!model) setVisionHint('Load a model in Settings to ask questions about the view.');
  else if (!isVlm) setVisionHint('The active model is text-only — load a vision (VLM) model to ask about images.');
}

// Shared by the Upload button and drag-and-drop: show an image file on the
// full-screen Vision stage and switch the source to it.
function loadVisionImageFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const image = document.getElementById('visionImage');
    if (image) image.src = e.target.result;
    setVisionSource('image');
  };
  reader.readAsDataURL(file);
}

function pickVisionImage() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.addEventListener('change', () => loadVisionImageFile(input.files && input.files[0]));
  input.click();
}

// Pull a still from the camera attached to the devkit board onto the stage.
async function snapVisionBoardCamera() {
  const btn = document.getElementById('visionBoardBtn');
  if (btn) btn.disabled = true;
  try {
    loadVisionImageFile(await fetchBoardCameraFrame());
  } catch (err) {
    console.error('Board camera snapshot failed:', err);
    setVisionSource('image');
    setVisionHint(`Board camera: ${err.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Capture the active vision source into a data URL, letterboxed to the model's
// input size (mirrors captureAndAnimateSnap so the backend path is identical).
function captureVisionSource(el) {
  if (!el) return null;
  const isVideo = el.tagName === 'VIDEO';
  const srcW = isVideo ? el.videoWidth : (el.naturalWidth || 0);
  const srcH = isVideo ? el.videoHeight : (el.naturalHeight || 0);
  if (!srcW || !srcH) return null;
  const visionSize = getVisionImageSize();
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (visionSize) {
    canvas.width = visionSize.width;
    canvas.height = visionSize.height;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const scale = Math.min(canvas.width / srcW, canvas.height / srcH);
    const dw = Math.max(1, Math.round(srcW * scale));
    const dh = Math.max(1, Math.round(srcH * scale));
    ctx.drawImage(el, 0, 0, srcW, srcH,
      Math.round((canvas.width - dw) / 2), Math.round((canvas.height - dh) / 2), dw, dh);
  } else {
    canvas.width = srcW;
    canvas.height = srcH;
    ctx.drawImage(el, 0, 0);
  }
  return canvas.toDataURL('image/png');
}

function askVision(query) {
  query = (query || '').trim();
  const input = document.getElementById('visionInput');
  if (!selectedChatModelSupportsVision()) {
    refreshVisionModelState();
    return;
  }
  const el = _visionSource === 'image'
    ? document.getElementById('visionImage')
    : document.getElementById('visionVideo');
  const captured = captureVisionSource(el);
  if (!captured) {
    setVisionHint(_visionSource === 'image' ? 'Upload an image first.' : 'The camera has no frame yet — give it a moment.');
    return;
  }
  if (!query) query = 'Describe what you see in the picture.';

  // Feed the existing send pipeline: it attaches snapAnimation.src as the image
  // when the image toggle is on, so force it on for a vision question.
  lastCapturedImageDataUrl = captured;
  snapAnimation.src = captured;
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');
  if (includeImageCheckbox) includeImageCheckbox.checked = true;

  const ans = document.getElementById('visionAnswer');
  if (ans) { ans.innerHTML = ''; ans.classList.remove('has-content'); }
  setVisionQuestion(query);
  if (typeof clearIfSingleShotMode === 'function') clearIfSingleShotMode();
  addChatMessage(query, true, true);
  startProcessing('', query, false);
  if (input) input.value = '';
}

// ---- Continuous camera loop: repeatedly ask the VLM about the live camera ----
let _visionLoop = { on: false, prompt: '', delayMs: 800 };
// Bumped on every start and stop. A startup that is mid-await when the
// loop is stopped (or restarted) sees a stale token and bows out, so it
// cannot clear a fresh conversation or race a second runner.
let _visionLoopToken = 0;

function waitForGenerationEnd(timeoutMs) {
  timeoutMs = timeoutMs || 90000;
  const start = performance.now();
  return new Promise((resolve) => {
    const tick = () => {
      if (!activeGeneration || (performance.now() - start) > timeoutMs || !_visionLoop.on) {
        return resolve();
      }
      setTimeout(tick, 150);
    };
    tick();
  });
}

// Each loop pass is a fresh look at the world, so drop the conversation once its
// answer lands. The server appends a user message per iteration — a base64 frame
// included — plus the reply, and never trims, so without this a running loop
// grows history without bound: memory climbs on the board, and with "Include
// chat history" on the prompt walks into the context limit within a few frames.
async function clearLoopContext() {
  chatHistory = [];
  try {
    await fetch('/clear-history', { method: 'POST' });
  } catch (err) {
    // Non-fatal: the next iteration still asks, it just carries the old turns.
    console.warn('Could not clear context between loop iterations:', err);
  }
}

function setVisionLoopUI(on) {
  const btn = document.getElementById('visionLoopBtn');
  if (!btn) return;
  btn.classList.toggle('is-looping', on);
  btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  btn.textContent = on ? '■ Stop' : '↻ Loop';
}

async function visionLoopRun(token) {
  while (_visionLoop.on && token === _visionLoopToken) {
    // The loop only makes sense on the live camera with a vision model.
    if (!isVisionOpen() || _visionSource !== 'camera' || !selectedChatModelSupportsVision()) {
      stopVisionLoop();
      break;
    }
    if (!activeGeneration) {
      const input = document.getElementById('visionInput');
      const p = (input && input.value.trim()) || _visionLoop.prompt || 'Describe what you see in the picture.';
      _visionLoop.prompt = p;
      askVision(p);                      // captures a fresh frame + asks (clears input)
      if (input) input.value = p;        // keep the prompt visible + editable for next iter
    }
    // Let the request register as active, then wait for it to complete.
    await new Promise((r) => setTimeout(r, 500));
    await waitForGenerationEnd();
    // Only once the answer is really finished — waitForGenerationEnd also
    // returns on its 90s timeout, and clearing mid-stream would drop the
    // generation id out from under the streaming thread and cut the reply off.
    if (!activeGeneration) await clearLoopContext();
    if (!_visionLoop.on) break;
    await new Promise((r) => setTimeout(r, _visionLoop.delayMs));
  }
}

function startVisionLoop() {
  if (_visionLoop.on) return;
  if (_visionSource !== 'camera') setVisionSource('camera');
  if (!selectedChatModelSupportsVision()) { refreshVisionModelState(); return; }
  const input = document.getElementById('visionInput');
  _visionLoop.prompt = (input && input.value.trim()) || 'Describe what you see in the picture.';
  _visionLoop.on = true;
  setVisionLoopUI(true);
  setVisionAskHint('Looping — asking about the live camera continuously. Tap Stop to end.');
  startVisionLoopRun(++_visionLoopToken);
}

// Start from a clean slate, so the first frame is not judged against whatever
// was said before the loop began — but never clear while a reply is streaming:
// that drops the server's generation id, so the stream ends without its `end`
// event and the loop would then wait out waitForGenerationEnd's full timeout.
async function startVisionLoopRun(token) {
  if (activeGeneration) {
    try { await stop(true); } catch (e) { /* best effort */ }
    await waitForGenerationEnd(15000);
    if (token !== _visionLoopToken) return;   // stopped or restarted meanwhile
  }
  await clearLoopContext();
  if (token !== _visionLoopToken) return;
  visionLoopRun(token);
}

function stopVisionLoop() {
  _visionLoop.on = false;
  _visionLoopToken++;          // invalidate any startup still mid-await

  setVisionLoopUI(false);
  if (isVisionOpen()) setVisionAskHint('');
}

function toggleVisionLoop() {
  if (_visionLoop.on) stopVisionLoop();
  else startVisionLoop();
}

// ---- Vision voice input (primary): speak a question about the current frame --
// Reuses the composer's recording primitives (startRecording / stopRecording /
// mediaRecorder / audioBlob) rather than spinning up a parallel MediaRecorder.

function isVisionOpen() {
  const modal = document.getElementById('visionModal');
  return !!modal && modal.style.display !== 'none';
}

function toggleVisionMic() {
  if (_visionRecording) stopVisionRecording();
  else startVisionRecording();
}

// Restore the shared mic track to whatever mute state the composer expects.
function restoreComposerMic() {
  try { if (audioTracks.length > 0) audioTracks[0].enabled = !isMicrophoneMuted; } catch (e) { /* ignore */ }
}

function setVisionRecordingUI(recording) {
  const btn = document.getElementById('visionMicBtn');
  if (btn) {
    btn.classList.toggle('is-recording', recording);
    btn.setAttribute('aria-pressed', recording ? 'true' : 'false');
    btn.title = recording ? 'Listening… tap to stop' : 'Tap to speak your question';
  }
}

function setVisionAskHint(text) {
  const el = document.getElementById('visionAskHint');
  if (el) el.textContent = text || VISION_ASK_HINT;
}

// Show the spoken/typed question above the streamed answer.
function setVisionQuestion(text) {
  const el = document.getElementById('visionQuestion');
  if (!el) return;
  const q = (text || '').trim();
  el.textContent = q ? `You asked: “${q}”` : '';
  el.classList.toggle('has-content', !!q);
}

function startVisionRecording() {
  if (_visionRecording) return;
  if (!selectedChatModelSupportsVision()) { refreshVisionModelState(); return; }
  if (!mediaStream || mediaStream.getAudioTracks().length === 0) {
    setVisionHint('No microphone available — type your question below instead.');
    return;
  }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    // Another recording (e.g. the composer mic) is already active — don't hijack it.
    setVisionHint('A recording is already in progress — try again in a moment.');
    return;
  }
  // The composer keeps the mic muted by default; enable it just for this capture.
  try { mediaStream.getAudioTracks().forEach((t) => { t.enabled = true; }); } catch (e) { /* ignore */ }
  startRecording();   // shared primitive: builds mediaRecorder from the audio tracks
  if (!mediaRecorder || mediaRecorder.state !== 'recording') {
    restoreComposerMic();
    setVisionHint('Could not access the microphone — type your question below instead.');
    return;
  }
  // Route this recorder's stop to the vision path instead of captureAndAnimateSnap.
  mediaRecorder.onstop = saveVisionRecording;
  _visionRecording = true;
  setVisionRecordingUI(true);
  setVisionAskHint('Listening… tap the mic to stop.');
}

// User tapped the mic to finish — stop and submit the spoken question.
function stopVisionRecording() {
  if (!_visionRecording) return;
  _visionRecording = false;
  setVisionRecordingUI(false);
  setVisionAskHint('Transcribing your question…');
  stopRecording();   // fires mediaRecorder.onstop -> saveVisionRecording
}

// Closing / bailing out — stop recording WITHOUT submitting a query.
function cancelVisionRecording() {
  if (!_visionRecording) return;
  _visionRecording = false;
  setVisionRecordingUI(false);
  setVisionAskHint('');
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.onstop = () => { recordedChunks = []; restoreComposerMic(); };
    try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
  } else {
    restoreComposerMic();
  }
  audioBlob = null;
}

// mediaRecorder.onstop handler for the submit path.
function saveVisionRecording() {
  try {
    audioBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
  } catch (e) {
    audioBlob = null;
  }
  restoreComposerMic();
  submitVisionVoiceQuery();
}

// Capture the current vision frame + send it with the recorded audio, exactly
// like askVision but on the audio path (server transcribes, then streams).
function submitVisionVoiceQuery() {
  setVisionAskHint('');
  if (!audioBlob) { setVisionHint('No audio was captured — tap the mic and try again.'); return; }
  const el = _visionSource === 'image'
    ? document.getElementById('visionImage')
    : document.getElementById('visionVideo');
  const captured = captureVisionSource(el);
  if (!captured) {
    audioBlob = null;
    setVisionHint(_visionSource === 'image' ? 'Upload an image first.' : 'The camera has no frame yet — give it a moment.');
    return;
  }
  // Feed the existing send pipeline: it attaches snapAnimation.src as the image
  // when the image toggle is on, so force it on for a vision question.
  lastCapturedImageDataUrl = captured;
  snapAnimation.src = captured;
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');
  if (includeImageCheckbox) includeImageCheckbox.checked = true;

  const ans = document.getElementById('visionAnswer');
  if (ans) { ans.innerHTML = ''; ans.classList.remove('has-content'); }
  setVisionQuestion('');   // filled in when the transcript arrives
  if (typeof clearIfSingleShotMode === 'function') clearIfSingleShotMode();
  // Audio + image path: waitForTranscription=true so the transcript comes back
  // via the 'transcription' socket event (displayTranscribedQuery).
  startProcessing('', null, true);
}

// Mirror the newest assistant answer from the chat into the vision overlay so
// streamed responses are visible without leaving full-screen.
function startVisionMirror() {
  const chat = document.getElementById('chatMessages');
  const ans = document.getElementById('visionAnswer');
  if (!chat || !ans || _visionMirrorObserver) return;
  const sync = () => {
    const assistants = chat.querySelectorAll('.message.assistant');
    const last = assistants[assistants.length - 1];
    if (!last) { ans.innerHTML = ''; ans.classList.remove('has-content'); return; }
    ans.innerHTML = last.innerHTML;
    // Strip cloned interactive bits (audio canvas, copy buttons) — read-only here.
    ans.querySelectorAll('canvas, button').forEach((n) => n.remove());
    ans.classList.add('has-content');
    // The mirror just replaced innerHTML — reattach the TTS highlight here.
    if (_ttsHi.container === ans) applyTtsHighlight();
    ans.scrollTop = ans.scrollHeight;
  };
  _visionMirrorObserver = new MutationObserver(sync);
  _visionMirrorObserver.observe(chat, { childList: true, subtree: true, characterData: true });
  sync();
}

function stopVisionMirror() {
  if (_visionMirrorObserver) { _visionMirrorObserver.disconnect(); _visionMirrorObserver = null; }
}

// ---- Full-screen Benchmark (web MoLE `perf`: TTFT / TPS) ---------------
let _benchPoll = null;

// GUI shutdown: stops the whole studio (UI + model server) via the supervisor.
function initShutdownButton() {
  const btn = document.getElementById('shutdownButton');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    if (!window.confirm('Shut down the GenAI Studio?\n\nThis stops the web app and the model server on the board. You will need to run ./run.sh again to restart it.')) return;
    btn.disabled = true;
    try { if (window.speechSynthesis) window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
    try {
      await fetch('/shutdown', { method: 'POST' });
    } catch (e) { /* the server is going down — a dropped request is expected */ }
    const overlay = document.getElementById('shutdownOverlay');
    if (overlay) overlay.style.display = 'flex';
  });
}

// ---- Showcase: the Modalix + LLiMa story, embedded full-screen in-app ----
// The header prism button opens the deck (showcase.html) inside an iframe overlay
// so it feels like a native mode (Vision/Benchmark) instead of navigating away.
// Modifier-clicks and middle-clicks still open /showcase in a new tab.
let _showcaseEntered = false;   // did we request browser fullscreen on open?

function initShowcase() {
  const modal = document.getElementById('showcaseModal');
  const btn = document.getElementById('showcaseButton');
  if (!modal || !btn) return;
  const close = document.getElementById('showcaseCloseBtn');
  btn.addEventListener('click', (e) => {
    // Let the browser handle new-tab intents (ctrl/cmd/shift/alt/middle-click).
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || (e.button && e.button !== 0)) return;
    e.preventDefault();
    openShowcase();
  });
  if (close) close.addEventListener('click', closeShowcase);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeShowcase();
  });
  // The deck runs inside the iframe; when it has focus, Escape can't reach us,
  // so showcase.html posts this message up so Esc still closes the overlay.
  window.addEventListener('message', (e) => {
    if (e && e.origin === window.location.origin && e.data === 'neat-showcase-close') closeShowcase();
  });
}

function openShowcase() {
  const modal = document.getElementById('showcaseModal');
  const frame = document.getElementById('showcaseFrame');
  if (!modal || !frame) return;
  // (Re)load fresh each time so the deck resets to slide 1 with autoplay stopped.
  frame.src = '/showcase';
  modal.style.display = 'flex';
  document.body.classList.add('showcase-open');
  _showcaseEntered = false;
  try {
    const rf = modal.requestFullscreen || modal.webkitRequestFullscreen;
    if (rf) { const p = rf.call(modal); if (p && p.then) { _showcaseEntered = true; p.catch(() => { _showcaseEntered = false; }); } }
  } catch (e) { /* ignore */ }
  // Hand keyboard control to the deck so arrows/dots/autoplay work immediately.
  setTimeout(() => { try { frame.contentWindow && frame.contentWindow.focus(); } catch (e) { /* ignore */ } }, 80);
}

function closeShowcase() {
  const modal = document.getElementById('showcaseModal');
  const frame = document.getElementById('showcaseFrame');
  if (!modal) return;
  modal.style.display = 'none';
  document.body.classList.remove('showcase-open');
  // Unload the deck so autoplay timers/audio stop and state is clean next open.
  if (frame) frame.src = 'about:blank';
  if (_showcaseEntered) {
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        const p = (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        if (p && p.catch) p.catch(() => {});
      }
    } catch (e) { /* ignore */ }
  }
  _showcaseEntered = false;
}

// ---- Solutions: SiMaSentry harness suites (Med/Safe/Sec), embedded in-app ----
// The header shield button opens a Studio-styled launcher grid; picking a card
// loads the vendored harness (/solutions/<mode>/) in a fullscreen iframe,
// pre-wired to the loaded model through the same-origin /v1/chat/completions
// proxy (the HTTPS page can't call the HTTP :9998 model server directly).
// The harness's own ⌂ Home action posts {type:'sima-sentry:home'} → back to grid.
const SOLUTIONS_MODES = {
  health:   { label: 'SiMaSentry-Med' },
  safety:   { label: 'SiMaSentry-Safe' },
  security: { label: 'SiMaSentry-Sec' },
};
let _solutionsEntered = false;   // did we request browser fullscreen on open?

function buildSolutionsHarnessUrl(mode) {
  // provider=ollama keeps the harness from requiring an API key; URL params
  // override its localStorage so every open reflects the current model.
  const params = new URLSearchParams({ provider: 'ollama', base_url: '/v1/chat/completions' });
  const model = getSelectedChatModel();
  if (model) params.set('model', model);
  return `/solutions/${mode}/index.html?${params.toString()}`;
}

function initSolutions() {
  const modal = document.getElementById('solutionsModal');
  const btn = document.getElementById('solutionsButton');
  if (!modal || !btn) return;
  btn.addEventListener('click', (e) => {
    // Let the browser handle new-tab intents (ctrl/cmd/shift/alt/middle-click).
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || (e.button && e.button !== 0)) return;
    e.preventDefault();
    openSolutions();
  });
  modal.querySelectorAll('.solutions-card').forEach((card) => {
    card.addEventListener('click', () => openSolutionsHarness(card.dataset.mode));
  });
  const home = document.getElementById('solutionsHomeBtn');
  if (home) home.addEventListener('click', showSolutionsGrid);
  const close = document.getElementById('solutionsCloseBtn');
  if (close) close.addEventListener('click', closeSolutions);
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || modal.style.display === 'none') return;
    const frame = document.getElementById('solutionsFrame');
    const frameVisible = frame && frame.style.display !== 'none';
    frameVisible ? showSolutionsGrid() : closeSolutions();
  });
  // Harnesses post {type:'sima-sentry:home'} when their Home action is used
  // while embedded — return to the launcher grid rather than closing outright.
  // (The showcase listener compares e.data to a string, so it ignores this.)
  window.addEventListener('message', (e) => {
    const frame = document.getElementById('solutionsFrame');
    if (!frame || e.source !== frame.contentWindow) return;
    if (e.origin !== window.location.origin) return;   // vendored ⇒ same origin
    if (e.data && typeof e.data === 'object' && e.data.type === 'sima-sentry:home') {
      showSolutionsGrid();
    }
  });
}

function updateSolutionsBadges() {
  const model = getSelectedChatModel();
  const vision = model && selectedChatModelSupportsVision();
  document.querySelectorAll('#solutionsModal .solutions-badge').forEach((badge) => {
    badge.hidden = !!vision;
    if (!model) badge.textContent = 'no model loaded';
    else if (!vision) badge.textContent = 'no vision support';
  });
  const note = document.getElementById('solutionsModelNote');
  if (note) {
    note.innerHTML = model
      ? `Using <b></b>${vision ? '' : ' — load a vision model for image features'}`
      : 'No model loaded — open Settings to load one';
    const slot = note.querySelector('b');
    if (slot) slot.textContent = model;
  }
}

function openSolutions() {
  const modal = document.getElementById('solutionsModal');
  if (!modal) return;
  updateSolutionsBadges();
  modal.style.display = 'flex';
  document.body.classList.add('solutions-open');
  _solutionsEntered = false;
  try {
    const rf = modal.requestFullscreen || modal.webkitRequestFullscreen;
    if (rf) { const p = rf.call(modal); if (p && p.then) { _solutionsEntered = true; p.catch(() => { _solutionsEntered = false; }); } }
  } catch (e) { /* ignore */ }
}

function openSolutionsHarness(mode) {
  if (!SOLUTIONS_MODES[mode]) return;
  const model = getSelectedChatModel();
  const vision = model && selectedChatModelSupportsVision();
  if (!vision) {
    const msg = model
      ? `${model} has no vision support — the ${SOLUTIONS_MODES[mode].label} image features won't work. Open anyway?`
      : `No model is loaded — ${SOLUTIONS_MODES[mode].label} cannot chat until one is loaded in Settings. Open anyway?`;
    if (!window.confirm(msg)) return;
  }
  const frame = document.getElementById('solutionsFrame');
  const grid = document.getElementById('solutionsGrid');
  if (!frame || !grid) return;
  const url = buildSolutionsHarnessUrl(mode);
  frame.src = url;
  const newTab = document.getElementById('solutionsNewTab');
  if (newTab) newTab.href = url;
  grid.style.display = 'none';
  frame.style.display = 'block';
  const home = document.getElementById('solutionsHomeBtn');
  if (home) home.style.display = 'inline-flex';
  setTimeout(() => { try { frame.contentWindow && frame.contentWindow.focus(); } catch (e) { /* ignore */ } }, 80);
}

function showSolutionsGrid() {
  const frame = document.getElementById('solutionsFrame');
  const grid = document.getElementById('solutionsGrid');
  if (frame) {
    // Unloading the iframe releases camera/mic and aborts any in-flight chat
    // fetch (the proxy then posts /stop so the accelerator is freed).
    frame.removeAttribute('src');
    frame.style.display = 'none';
  }
  if (grid) grid.style.display = 'flex';
  const home = document.getElementById('solutionsHomeBtn');
  if (home) home.style.display = 'none';
  const newTab = document.getElementById('solutionsNewTab');
  if (newTab) newTab.href = '/solutions/';
  updateSolutionsBadges();
}

function closeSolutions() {
  const modal = document.getElementById('solutionsModal');
  if (!modal) return;
  showSolutionsGrid();
  modal.style.display = 'none';
  document.body.classList.remove('solutions-open');
  if (_solutionsEntered) {
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        const p = (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        if (p && p.catch) p.catch(() => {});
      }
    } catch (e) { /* ignore */ }
  }
  _solutionsEntered = false;
}

function initBenchmark() {
  const modal = document.getElementById('benchmarkModal');
  if (!modal) return;
  const open = document.getElementById('benchmarkButton');
  const close = document.getElementById('benchmarkCloseBtn');
  const run = document.getElementById('benchRunBtn');
  const stop = document.getElementById('benchStopBtn');
  if (open) open.addEventListener('click', openBenchmark);
  // Phone drawer entry (the header button is hidden ≤480px): close the drawer
  // first so the modal isn't behind it.
  const openRail = document.getElementById('railBenchmarkButton');
  if (openRail) openRail.addEventListener('click', () => {
    const ws = document.querySelector('.workspace');
    if (ws) ws.classList.remove('rail-open');
    const rt = document.getElementById('railToggle');
    if (rt) rt.textContent = '☰';
    openBenchmark();
  });
  if (close) close.addEventListener('click', closeBenchmark);
  if (run) run.addEventListener('click', runBenchmark);
  if (stop) stop.addEventListener('click', stopBenchmark);
  const csv = document.getElementById('benchExportCsv');
  const json = document.getElementById('benchExportJson');
  if (csv) csv.addEventListener('click', exportBenchCsv);
  if (json) json.addEventListener('click', exportBenchJson);
  initBenchCombo();
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeBenchmark();
  });
}

function openBenchmark() {
  const modal = document.getElementById('benchmarkModal');
  if (!modal) return;
  modal.style.display = 'flex';
  document.body.classList.add('bench-open');
  populateBenchModels();   // list every available model to choose from
  pollBenchmarkOnce();     // reflect any in-progress or last-completed run
  try {
    const rf = modal.requestFullscreen || modal.webkitRequestFullscreen;
    if (rf) { const p = rf.call(modal); if (p && p.catch) p.catch(() => {}); }
  } catch (e) { /* ignore */ }
}

// ---- Benchmark: multi-model selection, sequential runs, comparison, export ----
// The "Models to benchmark" picker is a multi-select dropdown: check one, several,
// or all downloaded models. Running benchmarks each selected model in turn (loading
// it first) and then shows a side-by-side comparison that can be exported.
let _benchModels = [];              // catalog models (chat/VLM) shown in the picker
let _benchSelected = new Set();     // model names checked to benchmark
let _benchComparison = [];          // [{model, summary, runs, loadFailed}] this session
let _benchAbort = false;            // set by Stop to break the multi-model loop

async function populateBenchModels() {
  const trigger = document.getElementById('benchModelTrigger');
  if (!trigger) return;
  let catalog = [];
  try {
    if (controlEnabled()) {
      const r = await fetch('/models/catalog');
      const d = await r.json();
      catalog = (d && d.catalog) || [];
    } else {
      catalog = getConfiguredChatModels().map(name => ({ name, loaded: true, type: 'chat' }));
    }
  } catch (e) { catalog = []; }
  _benchModels = catalog.filter(m => (m.type || 'chat') !== 'asr');
  // Keep any prior selection that still exists; default to the active model.
  const names = new Set(_benchModels.map(m => m.name));
  _benchSelected = new Set(Array.from(_benchSelected).filter(n => names.has(n)));
  if (!_benchSelected.size) {
    const active = getSelectedChatModel();
    if (active && names.has(active)) _benchSelected.add(active);
  }
  renderBenchModelList(document.getElementById('benchModelSearch')?.value || '');
  updateBenchModelSummary();
  updateBenchModelState();
}

function renderBenchModelList(filter) {
  const list = document.getElementById('benchModelList');
  if (!list) return;
  const f = (filter || '').trim().toLowerCase();
  const items = _benchModels.filter(m => !f || m.name.toLowerCase().includes(f));
  list.innerHTML = '';
  if (!items.length) {
    list.innerHTML = '<div class="bench-model-empty">No downloaded models match.</div>';
    return;
  }
  items.forEach(m => {
    const label = document.createElement('label');
    label.className = 'bench-model-opt';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.value = m.name; cb.checked = _benchSelected.has(m.name);
    cb.addEventListener('change', () => {
      if (cb.checked) _benchSelected.add(m.name); else _benchSelected.delete(m.name);
      updateBenchModelSummary(); updateBenchModelState();
    });
    const txt = document.createElement('span');
    const size = m.sizeBytes ? `  ·  ${fmtBytes(m.sizeBytes)}` : '';
    txt.textContent = `${m.loaded ? '● ' : ''}${m.name}  ·  ${typeBadge(m.type)}${size}`;
    label.appendChild(cb); label.appendChild(txt);
    list.appendChild(label);
  });
}

// Selected model names in catalog order (only ones that still exist).
function benchSelectedModels() {
  return _benchModels.map(m => m.name).filter(n => _benchSelected.has(n));
}

function updateBenchModelSummary() {
  const summary = document.getElementById('benchModelSummary');
  const all = document.getElementById('benchModelAll');
  const total = _benchModels.length;
  const sel = benchSelectedModels();
  const n = sel.length;
  if (summary) {
    summary.textContent = n === 0 ? 'Select models…'
      : (total && n === total ? `All ${total} models`
        : (n === 1 ? sel[0] : `${n} models`));
  }
  if (all) { all.checked = total > 0 && n === total; all.indeterminate = n > 0 && n < total; }
}

function openBenchMenu() {
  const menu = document.getElementById('benchModelMenu');
  const trigger = document.getElementById('benchModelTrigger');
  if (!menu) return;
  renderBenchModelList(document.getElementById('benchModelSearch')?.value || '');
  menu.hidden = false;
  if (trigger) trigger.setAttribute('aria-expanded', 'true');
}
function closeBenchMenu() {
  const menu = document.getElementById('benchModelMenu');
  const trigger = document.getElementById('benchModelTrigger');
  if (menu) menu.hidden = true;
  if (trigger) trigger.setAttribute('aria-expanded', 'false');
}

function initBenchCombo() {
  const trigger = document.getElementById('benchModelTrigger');
  const menu = document.getElementById('benchModelMenu');
  const search = document.getElementById('benchModelSearch');
  const all = document.getElementById('benchModelAll');
  if (!trigger || !menu) return;
  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (menu.hidden) openBenchMenu(); else closeBenchMenu();
  });
  if (search) search.addEventListener('input', () => renderBenchModelList(search.value));
  if (all) all.addEventListener('change', () => {
    if (all.checked) _benchModels.forEach(m => _benchSelected.add(m.name));
    else _benchSelected.clear();
    renderBenchModelList(search ? search.value : '');
    updateBenchModelSummary(); updateBenchModelState();
  });
  document.addEventListener('click', (e) => {
    const combo = document.getElementById('benchModelCombo');
    if (combo && !combo.contains(e.target)) closeBenchMenu();
  });
}

// "Model N of M: name" while a multi-model run is in flight.
function setBenchScope(idx, total, model) {
  const el = document.getElementById('benchScope');
  if (!el) return;
  if (total <= 1) { el.style.display = 'none'; return; }
  el.style.display = 'block';
  el.textContent = `Benchmarking model ${idx + 1} of ${total}: ${model}`;
}
function clearBenchScope() {
  const el = document.getElementById('benchScope');
  if (el) el.style.display = 'none';
}

// Poll /benchmark/status (rendering live) until the current model's run finishes.
function pollBenchmarkUntilDone() {
  return new Promise((resolve) => {
    const tick = async () => {
      if (_benchAbort) {
        // Make sure any run that DID start server-side is actually stopped, so it
        // doesn't keep occupying the accelerator after we abort.
        try { await fetch('/benchmark/stop', { method: 'POST' }); } catch (e) { /* ignore */ }
        resolve({ summary: null, runs: [] });
        return;
      }
      let d = null;
      try { const r = await fetch('/benchmark/status'); d = await r.json(); } catch (e) { d = null; }
      if (d) renderBenchmark(d);
      if (d && !d.running) { resolve(d); return; }
      setTimeout(tick, 500);
    };
    tick();
  });
}

// Load (if needed) + benchmark ONE model; returns {summary, runs, loadFailed}.
async function runOneBenchmark(model, runs, maxTokens, prompt) {
  const hint = document.getElementById('benchHint');
  if (controlEnabled() && getSelectedChatModel() !== model) {
    showBenchLoadPanel(model);
    beginLoadMirror('benchLoadStatus', 'benchLoadLog', 'benchLoadBar');
    try { await loadModelAndActivate(model); } finally { endLoadMirror(); }
    hideBenchLoadPanel();
    if (getSelectedChatModel() !== model) return { summary: null, runs: [], loadFailed: true };
  }
  // Stop pressed during the (multi-second) load — don't start a server run that
  // would then be orphaned.
  if (_benchAbort) return { summary: null, runs: [] };
  let started;
  try {
    const resp = await fetch('/benchmark/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_samples: runs, max_new_tokens: maxTokens, prompt, model }),
    });
    started = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(started.error || 'failed to start');
  } catch (err) {
    if (hint) hint.textContent = `Could not benchmark ${model}: ${err.message}`;
    return { summary: null, runs: [] };
  }
  renderBenchmark(started);
  const final = await pollBenchmarkUntilDone();
  return { summary: (final && final.summary) || null, runs: (final && final.runs) || [] };
}

// Side-by-side comparison table + bar chart (shown once 2+ models are involved).
function renderBenchComparison(list) {
  const wrap = document.getElementById('benchCompare');
  const body = document.getElementById('benchCompareBody');
  if (!wrap || !body) return;
  if (list.length < 2) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  const done = list.filter(e => e.summary);
  const bestTps = done.length ? Math.max(...done.map(e => e.summary.tps.mean)) : null;
  const bestTtft = done.length ? Math.min(...done.map(e => e.summary.ttftMs.mean)) : null;
  body.innerHTML = '';
  list.forEach(e => {
    const tr = document.createElement('tr');
    const cell = (txt, cls) => { const td = document.createElement('td'); td.textContent = txt; if (cls) td.className = cls; return td; };
    tr.appendChild(cell(e.model, 'bench-compare-model'));
    const s = e.summary;
    if (!s) {
      const td = document.createElement('td');
      td.colSpan = 6; td.className = 'bench-compare-fail';
      td.textContent = e.loadFailed ? 'failed to load' : 'no valid result';
      tr.appendChild(td); body.appendChild(tr); return;
    }
    const tpsTd = cell(benchFmt(s.tps.mean)); if (s.tps.mean === bestTps) tpsTd.classList.add('bench-best');
    const ttftTd = cell(benchFmt(s.ttftMs.mean)); if (s.ttftMs.mean === bestTtft) ttftTd.classList.add('bench-best');
    tr.appendChild(tpsTd);
    tr.appendChild(cell(benchFmt(s.tps.p90)));
    tr.appendChild(ttftTd);
    tr.appendChild(cell(benchFmt(s.tokens.mean)));
    tr.appendChild(cell(benchFmt(s.tps.stdev)));
    tr.appendChild(cell(`${s.count}${s.errors ? ' · ' + s.errors + ' err' : ''}`));
    body.appendChild(tr);
  });
  renderBenchCompareChart(list);
}

function renderBenchCompareChart(list) {
  const chart = document.getElementById('benchCompareChart');
  if (!chart) return;
  chart.innerHTML = '';
  const done = list.filter(e => e.summary && e.summary.tps);
  if (!done.length) return;
  const maxT = Math.max(1, ...done.map(e => e.summary.tps.mean || 0));
  const best = Math.max(...done.map(e => e.summary.tps.mean || 0));
  done.forEach(e => {
    const tps = e.summary.tps.mean || 0;
    const row = document.createElement('div'); row.className = 'bench-cbar-row';
    const name = document.createElement('span'); name.className = 'bench-cbar-name';
    name.textContent = e.model; name.title = e.model;
    const track = document.createElement('div'); track.className = 'bench-cbar-track';
    const fill = document.createElement('div'); fill.className = 'bench-cbar-fill' + (tps === best ? ' best' : '');
    fill.style.width = Math.max(2, Math.round(tps / maxT * 100)) + '%';
    const val = document.createElement('span'); val.className = 'bench-cbar-val'; val.textContent = benchFmt(tps) + ' tok/s';
    track.appendChild(fill);
    row.appendChild(name); row.appendChild(track); row.appendChild(val);
    chart.appendChild(row);
  });
}

// ---- Export -----------------------------------------------------------
function benchExportStamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function downloadBlob(filename, text, type) {
  try {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  } catch (e) { console.warn('export failed', e); }
}

function benchExportRows() {
  return _benchComparison.filter(e => e.summary).map(e => ({
    model: e.model,
    tpsMean: e.summary.tps.mean, tpsMedian: e.summary.tps.median, tpsMin: e.summary.tps.min,
    tpsMax: e.summary.tps.max, tpsP90: e.summary.tps.p90, tpsStdev: e.summary.tps.stdev,
    ttftMeanMs: e.summary.ttftMs.mean, ttftP90Ms: e.summary.ttftMs.p90,
    tokensMean: e.summary.tokens.mean, validRuns: e.summary.count, errors: e.summary.errors,
  }));
}

function exportBenchCsv() {
  const rows = benchExportRows();
  if (!rows.length) return;
  const cols = ['model', 'tpsMean', 'tpsMedian', 'tpsMin', 'tpsMax', 'tpsP90', 'tpsStdev',
    'ttftMeanMs', 'ttftP90Ms', 'tokensMean', 'validRuns', 'errors'];
  const esc = (v) => { const s = String(v == null ? '' : v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const lines = [cols.join(',')].concat(rows.map(r => cols.map(c => esc(r[c])).join(',')));
  downloadBlob(`neat-benchmark-${benchExportStamp()}.csv`, lines.join('\n'), 'text/csv');
}

function exportBenchJson() {
  if (!_benchComparison.some(e => e.summary)) return;
  const payload = {
    generatedAt: new Date().toISOString(),
    config: {
      runs: benchClampInt('benchRuns', 1, 50, 5),
      maxNewTokens: benchClampInt('benchMaxTokens', 8, 2048, 128),
      prompt: (document.getElementById('benchPrompt') || {}).value || '(default)',
    },
    results: _benchComparison.map(e => ({
      model: e.model, summary: e.summary, runs: e.runs, loadFailed: !!e.loadFailed,
    })),
  };
  downloadBlob(`neat-benchmark-${benchExportStamp()}.json`, JSON.stringify(payload, null, 2), 'application/json');
}

function closeBenchmark() {
  const modal = document.getElementById('benchmarkModal');
  if (!modal) return;
  modal.style.display = 'none';
  document.body.classList.remove('bench-open');
  // Abort any in-flight multi-model run so it doesn't keep loading models and
  // benchmarking in the background (the Stop button leaves with the modal).
  const stopBtn = document.getElementById('benchStopBtn');
  if (stopBtn && stopBtn.style.display !== 'none') stopBenchmark();
  stopBenchPolling();
  endLoadMirror();
  hideBenchLoadPanel();
  try {
    if (document.fullscreenElement || document.webkitFullscreenElement) {
      const p = (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      if (p && p.catch) p.catch(() => {});
    }
  } catch (e) { /* ignore */ }
}

function updateBenchModelState() {
  const run = document.getElementById('benchRunBtn');
  const hint = document.getElementById('benchHint');
  const n = benchSelectedModels().length;
  const busy = serverBusy();
  if (run) {
    run.disabled = n === 0 || busy;
    run.textContent = n > 1 ? `Run benchmark · ${n} models` : 'Run benchmark';
  }
  if (hint) {
    hint.textContent = n ? ''
      : (_benchModels.length ? 'Select one or more models to benchmark.'
        : 'No models available — add or download one in Settings → Models.');
  }
}

function benchClampInt(id, min, max, def) {
  const el = document.getElementById(id);
  let v = parseInt(el && el.value, 10);
  if (!Number.isFinite(v)) v = def;
  v = Math.max(min, Math.min(max, v));
  if (el) el.value = v;
  return v;
}
function benchFmt(n) { return (n == null) ? '–' : (Math.round(n * 100) / 100).toLocaleString(); }

async function runBenchmark() {
  const models = benchSelectedModels();
  const hint = document.getElementById('benchHint');
  if (!models.length || _modelBusy) { updateBenchModelState(); return; }
  const runs = benchClampInt('benchRuns', 1, 50, 5);
  const maxTokens = benchClampInt('benchMaxTokens', 8, 2048, 128);
  const promptEl = document.getElementById('benchPrompt');
  const prompt = promptEl ? promptEl.value : '';
  if (hint) hint.textContent = '';
  _benchComparison = [];
  _benchAbort = false;
  renderBenchComparison(_benchComparison);   // clear any prior comparison
  applyBenchRunning(true);
  const total = models.length;
  for (let i = 0; i < total; i++) {
    if (_benchAbort) break;
    const model = models[i];
    setBenchScope(i, total, model);
    const res = await runOneBenchmark(model, runs, maxTokens, prompt);
    _benchComparison.push({ model, summary: res.summary, runs: res.runs, loadFailed: !!res.loadFailed });
    renderBenchComparison(_benchComparison);   // update the table/chart as each finishes
  }
  clearBenchScope();
  applyBenchRunning(false);
  if (_benchAbort && hint) hint.textContent = 'Benchmark stopped.';
}

async function stopBenchmark() {
  _benchAbort = true;                         // break the multi-model loop
  try { await fetch('/benchmark/stop', { method: 'POST' }); } catch (e) { /* ignore */ }
}

function startBenchPolling() { stopBenchPolling(); _benchPoll = setInterval(pollBenchmarkOnce, 500); }
function stopBenchPolling() { if (_benchPoll) { clearInterval(_benchPoll); _benchPoll = null; } }

async function pollBenchmarkOnce() {
  let d = null;
  try { const r = await fetch('/benchmark/status'); d = await r.json(); } catch (e) { return; }
  if (!d) return;
  applyBenchRunning(!!d.running);
  renderBenchmark(d);
  if (!d.running) stopBenchPolling();
}

function applyBenchRunning(running) {
  const run = document.getElementById('benchRunBtn');
  const stop = document.getElementById('benchStopBtn');
  if (run) run.style.display = running ? 'none' : '';
  if (stop) stop.style.display = running ? '' : 'none';
  ['benchModelTrigger', 'benchRuns', 'benchMaxTokens', 'benchPrompt'].forEach(id => {
    const el = document.getElementById(id); if (el) el.disabled = !!running;
  });
  if (running) closeBenchMenu();
  if (!running) updateBenchModelState();
}

// Live model-load progress inside the benchmark (fed by the shared load mirror).
function showBenchLoadPanel(name) {
  const panel = document.getElementById('benchLoadPanel');
  if (!panel) return;
  const title = document.getElementById('benchLoadTitle');
  if (title) title.textContent = `Loading ${name}…`;
  const status = document.getElementById('benchLoadStatus');
  if (status) { status.textContent = 'Preparing…'; status.className = 'bench-load-status loading'; }
  const log = document.getElementById('benchLoadLog');
  if (log) log.textContent = '';
  panel.style.display = 'block';
}
function hideBenchLoadPanel() {
  const panel = document.getElementById('benchLoadPanel');
  if (panel) panel.style.display = 'none';
}

function renderBenchmark(d) {
  const total = d.total || 0, done = d.done || 0;
  const prog = document.getElementById('benchProgress');
  const fill = document.getElementById('benchProgressFill');
  const ptext = document.getElementById('benchProgressText');
  if (prog) prog.style.display = (d.running || done) ? 'flex' : 'none';
  if (fill) fill.style.width = (total ? Math.round(done / total * 100) : 0) + '%';
  if (ptext) ptext.textContent = d.running ? `Running… ${done}/${total}` : (done ? `Done · ${done}/${total} runs` : '');

  // Prompt used (default or custom) — for transparency.
  const promptShown = document.getElementById('benchPromptShown');
  const promptText = document.getElementById('benchPromptText');
  if (promptShown && promptText) {
    if (d.prompt) { promptText.textContent = d.prompt; promptShown.style.display = 'flex'; }
    else promptShown.style.display = 'none';
  }
  // Live output of the in-flight run.
  const live = document.getElementById('benchLive');
  const cur = d.current;
  if (live) {
    if (cur) {
      live.style.display = 'block';
      const title = document.getElementById('benchLiveTitle');
      const stats = document.getElementById('benchLiveStats');
      const txt = document.getElementById('benchLiveText');
      if (title) title.textContent = `Generating · run ${(cur.index || 0) + 1}/${total}`;
      if (stats) stats.textContent = `${cur.tokens || 0} tokens`
        + (cur.ttftMs != null ? ` · TTFT ${benchFmt(cur.ttftMs)} ms` : '')
        + (cur.tps != null ? ` · ${benchFmt(cur.tps)} tok/s` : '');
      if (txt) {
        const atBottom = txt.scrollTop + txt.clientHeight >= txt.scrollHeight - 8;
        // Render the streamed output as markdown (same pipeline as the chat) so
        // headings, code, lists and math appear formatted, not raw.
        if (txt._raw !== (cur.text || '')) renderMarkdownStreaming(txt, cur.text || '');
        if (atBottom) txt.scrollTop = txt.scrollHeight;
      }
    } else {
      live.style.display = 'none';
    }
  }

  const s = d.summary;
  const summary = document.getElementById('benchSummary');
  if (s && summary) {
    summary.style.display = 'grid';
    document.getElementById('benchTtftMean').textContent = benchFmt(s.ttftMs.mean);
    document.getElementById('benchTtftSub').textContent = `median ${benchFmt(s.ttftMs.median)} · min ${benchFmt(s.ttftMs.min)} · max ${benchFmt(s.ttftMs.max)} · σ ${benchFmt(s.ttftMs.stdev)} ms` + (s.errors ? ` · ${s.errors} error${s.errors > 1 ? 's' : ''}` : '');
    document.getElementById('benchTpsMean').textContent = benchFmt(s.tps.mean);
    const srvMeasured = (d.runs || []).some(r => r && r.serverMetrics);
    document.getElementById('benchTpsSub').textContent = `median ${benchFmt(s.tps.median)} · min ${benchFmt(s.tps.min)} · max ${benchFmt(s.tps.max)} · σ ${benchFmt(s.tps.stdev)} tok/s · ${srvMeasured ? 'runtime-measured' : 'client-timed'}`;
  } else if (summary) {
    summary.style.display = 'none';
  }
  renderBenchStats(s);

  const runs = d.runs || [];
  const results = document.getElementById('benchResults');
  if (results) results.style.display = runs.length ? 'grid' : 'none';
  renderBenchChart(runs);
  renderBenchTable(runs);
  renderBenchResponses(runs);
}

function renderBenchResponses(runs) {
  const wrap = document.getElementById('benchResponses');
  const list = document.getElementById('benchResponsesList');
  if (!wrap || !list) return;
  const withText = runs.filter(r => r && !r.error && r.text);
  if (!withText.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  list.innerHTML = '';
  runs.forEach((r, i) => {
    if (!r || r.error || !r.text) return;
    const det = document.createElement('details');
    det.className = 'bench-response';
    const sum = document.createElement('summary');
    const run = document.createElement('span'); run.className = 'bench-response-run'; run.textContent = `Run ${i + 1}`;
    const meta = document.createElement('span'); meta.className = 'bench-response-meta';
    meta.textContent = `${benchFmt(r.tps)} tok/s · TTFT ${benchFmt(r.ttftMs)} ms · ${r.tokens} tokens`;
    sum.appendChild(run); sum.appendChild(meta);
    const body = document.createElement('div');
    body.className = 'bench-response-text message-text';
    renderMarkdownInto(body, r.text);   // format like the chat (code, lists, math)
    det.appendChild(sum); det.appendChild(body);
    list.appendChild(det);
  });
}

function renderBenchChart(runs) {
  const chart = document.getElementById('benchChart');
  if (!chart) return;
  chart.innerHTML = '';
  const maxT = Math.max(1, ...runs.map(r => (r && !r.error) ? (r.tps || 0) : 0));
  runs.forEach((r, i) => {
    const bar = document.createElement('div');
    bar.className = 'bench-bar' + (r && r.error ? ' err' : '');
    bar.style.height = (r && !r.error ? Math.max(4, Math.round((r.tps / maxT) * 100)) : 100) + '%';
    bar.title = (r && r.error) ? `Run ${i + 1}: error` : `Run ${i + 1}: ${benchFmt(r.tps)} tok/s · TTFT ${benchFmt(r.ttftMs)} ms`;
    chart.appendChild(bar);
  });
}

// Scientific summary: per-metric min / max / mean / median / std-dev / p90.
function renderBenchStats(s) {
  const wrap = document.getElementById('benchStats');
  const tb = document.getElementById('benchStatsBody');
  if (!wrap || !tb) return;
  if (!s) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  const rows = [
    ['TTFT (ms)', s.ttftMs],
    ['Throughput (tok/s)', s.tps],
    ['Tokens / run', s.tokens],
  ];
  tb.innerHTML = '';
  rows.forEach(([label, m]) => {
    if (!m) return;
    const tr = document.createElement('tr');
    const cells = [label, m.min, m.max, m.mean, m.median, m.stdev, m.p90];
    cells.forEach((v, i) => {
      const td = document.createElement('td');
      td.textContent = (i === 0) ? v : benchFmt(v);
      if (i === 0) td.className = 'bench-stats-metric';
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  const foot = document.getElementById('benchStatsFoot');
  if (foot) {
    const cv = (s.tps && s.tps.cv != null) ? `${benchFmt(s.tps.cv)}%` : '–';
    foot.textContent = `${s.count} valid run${s.count === 1 ? '' : 's'}`
      + (s.errors ? ` · ${s.errors} error${s.errors > 1 ? 's' : ''}` : '')
      + ` · ${s.totalTokens} tokens total · throughput CV ${cv}`;
  }
}

function renderBenchTable(runs) {
  const tb = document.getElementById('benchTableBody');
  if (!tb) return;
  tb.innerHTML = '';
  runs.forEach((r, i) => {
    const tr = document.createElement('tr');
    if (r && r.error) {
      const td0 = document.createElement('td'); td0.textContent = i + 1;
      const td1 = document.createElement('td'); td1.colSpan = 3; td1.className = 'bench-err'; td1.textContent = r.error;
      tr.appendChild(td0); tr.appendChild(td1);
    } else {
      [i + 1, benchFmt(r.ttftMs), benchFmt(r.tps), r.tokens].forEach(v => {
        const td = document.createElement('td'); td.textContent = v; tr.appendChild(td);
      });
    }
    tb.appendChild(tr);
  });
}
