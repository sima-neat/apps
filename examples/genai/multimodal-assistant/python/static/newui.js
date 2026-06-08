// New Dashboard Elements
const cameraPreview = document.getElementById('cameraPreview');
const snapAnimation = document.getElementById('snapAnimation');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const recordButton = document.getElementById('recordButton');
const recordIcon = document.getElementById('recordIcon');
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
let audioTracks = [];
let recordedChunks = [];

const socket = io('/');
const audioQueue = [];
let isPlaying = false;
let currentAudioContext = null;
let receivedEndSignal = false;
let shouldPlayAudio = true;
let currentSourceNode = null;
let activeGeneration = false;
let pendingNewGenerationAudio = false;
let requestQueue = Promise.resolve();

// First Audio timing tracking
let userInputStartTime = null;
let firstAudioStarted = false;

let ragServerStatusText = "";

let currentSystemPrompt = '';
let systemPromptRequestInFlight = false;

// Helper function to check if we're in LLM-only mode
function isLlmOnlyMode() {
  return document.body.classList.contains('llm-only');
}

function isRagEnabled() {
  const config = window.SIMA_CONFIG || {};
  return config.ragEnabled === true || config.ragEnabled === 'true';
}

function hideRagControlsIfDisabled() {
  if (isRagEnabled()) return;

  const ragContainer = document.querySelector('.settings-right-container');
  const ragCheckbox = document.getElementById('toggleRAG');

  if (ragContainer) {
    ragContainer.style.display = 'none';
  }
  if (ragCheckbox) {
    ragCheckbox.checked = false;
  }
}

function getVisionImageSize() {
  const config = window.SIMA_CONFIG || {};
  const height = parseInt(config.visionImageHeight, 10);
  const width = parseInt(config.visionImageWidth, 10);

  if (Number.isFinite(height) && Number.isFinite(width) && height > 0 && width > 0) {
    return { height, width };
  }

  return null;
}

// Access the local camera and microphone feed
async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      },
      audio: true
    });

    cameraPreview.srcObject = mediaStream;
    audioTracks = mediaStream.getAudioTracks();
    toggleMicrophone(true);

    // Set up dynamic camera container sizing
    setupCameraContainer();
  } catch (error) {
    console.error('Error accessing media devices.', error);
  }
}

async function startAudioOnly() {
  try {
    // Only request audio access, no video
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: true
    });

    audioTracks = mediaStream.getAudioTracks();
    toggleMicrophone(true);
    console.log('Audio-only mode initialized for LLM');
  } catch (error) {
    console.error('Error accessing audio devices.', error);
  }
}

// Dynamic camera container sizing
function setupCameraContainer() {
  const container = document.querySelector('.camera-preview-container');
  const cameraSection = document.getElementById('cameraSection');

  if (!container || !cameraSection || !cameraPreview) return;

  // Listen for video metadata to get natural dimensions
  cameraPreview.addEventListener('loadedmetadata', () => {
    const videoWidth = cameraPreview.videoWidth;
    const videoHeight = cameraPreview.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) return;

    // Get camera section dimensions
    const sectionRect = cameraSection.getBoundingClientRect();
    const sectionWidth = sectionRect.width - 40; // Account for padding/margins
    const sectionHeight = sectionRect.height - 40;

    // Calculate aspect ratio
    const videoAspect = videoWidth / videoHeight;

    // Calculate optimal container size that fits within camera section
    let containerWidth = sectionWidth;
    let containerHeight = containerWidth / videoAspect;

    // If height is too tall, scale by height instead
    if (containerHeight > sectionHeight) {
      containerHeight = sectionHeight;
      containerWidth = containerHeight * videoAspect;
    }

    // Apply minimum sizes
    containerWidth = Math.max(containerWidth, 200);
    containerHeight = Math.max(containerHeight, 150);

    // Update container dimensions
    container.style.width = `${containerWidth}px`;
    container.style.height = `${containerHeight}px`;
  });

  // Handle window resize
  window.addEventListener('resize', () => {
    if (cameraPreview.videoWidth > 0) {
      // Trigger metadata event to recalculate
      const event = new Event('loadedmetadata');
      cameraPreview.dispatchEvent(event);
    }
  });
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
  // Handle LLM-only mode setup - detect via CSS class
  const isLlmOnly = document.body.classList.contains('llm-only');

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

    // Audio-only mode for LLM
    startAudioOnly();
  } else {
    // Full camera + audio
    startCamera();
  }

  // Initialize dashboard functionality
  initializeVoiceSync();
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
    if (isLlmOnly) {
      // LLM-only mode: enable history by default
      chatHistoryCheckbox.checked = true;
      addChatMessage("Hi, this is the Neat Multi-Modal Assistant! How can I help you? Chat history is enabled.", false);
    } else {
      // VLM mode: disable history by default (multi-image history is slow)
      chatHistoryCheckbox.checked = false;
      addChatMessage("Hi, this is the Neat Multi-Modal Assistant! How can I help you? Chat history disabled. Enable 'Include chat history' in settings for multi-turn conversations.", false);
    }
    chatHistoryCheckbox.addEventListener('change', handleChatHistoryToggle);
  } else {
    // Fallback if checkbox doesn't exist
    addChatMessage("Hi, this is the Neat Multi-Modal Assistant! How can I help you?", false);
  }
};

function toggleMicrophone(mute) {
  const mutedUrl = recordButton.getAttribute('data-muted-url');
  const activeUrl = recordButton.getAttribute('data-active-url');

  recordIcon.src = mute ? mutedUrl : activeUrl;

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
  messageDiv.textContent = message;

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

    // Add info message
    addChatMessage("Hi, this is the Neat Multi-Modal Assistant! How can I help you? Chat history disabled. Enable 'Include chat history' in settings for multi-turn conversations.", false);
  } else {
    // Checkbox was re-enabled - keep user conversation, update info message if no conversation
    console.log('Chat history enabled - conversations will accumulate');

    // Check if there are any USER messages (actual conversation)
    const userMessages = chatMessages.querySelectorAll('.message.user');
    if (userMessages.length === 0) {
      // No user conversation - clear info messages and show new status
      const allMessages = chatMessages.querySelectorAll('.message');
      allMessages.forEach(message => message.remove());
      addChatMessage("Hi, this is the Neat Multi-Modal Assistant! How can I help you? Chat history enabled. Conversations will be remembered.", false);
    }
    // If there are user messages, keep them and don't add any info message
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

  // Re-add welcome message
  addChatMessage("Hi, this is the Neat Multi-Modal Assistant! How can I help you?", false);

  // Hide abort button if visible
  hideAbortButton();
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

  // Reset language select to English
  const languageSelect = document.getElementById('languageSelect');
  if (languageSelect) {
    languageSelect.value = 'en';
  }

  // Reset voice select to default
  const voiceSelect = document.getElementById('voiceSelect');
  if (voiceSelect) {
    voiceSelect.value = 'default';
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
  // Stop backend processing and audio
  stop(true);

  // Remove speaking indicator from current assistant message
  const assistantMessages = chatMessages.querySelectorAll('.message.assistant');
  const currentAssistantMessage = assistantMessages[assistantMessages.length - 1];
  if (currentAssistantMessage) {
    currentAssistantMessage.classList.remove('speaking');

    // Hide the audio visualizer
    currentAssistantMessage.classList.remove('audio-playing');
    const canvas = currentAssistantMessage.querySelector('.audio-visualizer');
    if (canvas) {
      canvas.style.display = 'none';
    }

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
  // Only show language row if URL parameter allows it
  const params = new URLSearchParams(window.location.search);
  const languageRow = document.getElementById('languageRow');

  if (languageRow && params.get('languageselect') === '1') {
    languageRow.style.display = 'block';
  }

  // Voice row logic depends on language selection
  const voiceRow = document.getElementById('voiceRow');
  const languageSelect = document.getElementById('languageSelect');
  if (voiceRow && languageSelect) {
    voiceRow.style.display = languageSelect.value === 'en' ? 'block' : 'none';
  }
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

  const sendRequest = async () => {
    activeGeneration = true;
    receivedEndSignal = false;
    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      displayResult(data.question || resultMessage, 'static/sample_audio.wav', data.ttt);
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
  shouldPlayAudio = false;
  isPlaying = false;
  audioQueue.length = 0;

  try {
    if (currentSourceNode) {
      currentSourceNode.stop(0);
      currentSourceNode.disconnect();
    }
  } catch (e) {
    console.warn("Error stopping source node:", e);
  } finally {
    currentSourceNode = null;
  }

  try {
    if (currentAudioContext) {
      currentAudioContext.close();
    }
  } catch (e) {
    console.warn("Error closing audio context:", e);
  } finally {
    currentAudioContext = null;
  }

  console.log("🧹 Audio playback completely stopped and cleaned up.");
}

function displayResult(text, audioSrc, ttt = 0) {
  // Only update metrics - text display is now handled by WebSocket events
  // The assistant message should already exist from startProcessing()
  document.getElementById('transcribeTime').textContent = ttt + 's';

  // Note: text parameter (data.question) is ignored since it's just echoing user input
  // Actual LLM response comes through handleTextUpdate() via WebSocket
}


function selectImage() {
  // Do nothing in LLM-only mode
  if (isLlmOnlyMode()) return;

  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';

  input.onchange = (event) => {
    const file = event.target.files[0];
    if (file) {
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
  };

  input.click();
}

socket.on('audio_chunk', (data) => {
  // Reject chunks if audio playback was aborted
  if (!shouldPlayAudio) {
    console.log('Ignoring audio chunk - playback aborted');
    return;
  }

  const text = data.text;
  const audioData = data.audio;

  console.log('Received text & audio :', text);

  document.getElementById('tpsValue').textContent = data.tps;
  document.getElementById('rtfValue').textContent = data.rtf;

  // Don't override shouldPlayAudio - respect abort state
  audioQueue.push(audioData);
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

    // Append text to the text container, not the message directly
    const textContainer = currentAssistantMessage.querySelector('.message-text') || currentAssistantMessage;
    if (textContainer.className === 'message-text') {
      textContainer.textContent += cleanText;
    } else {
      // Fallback for messages without text container
      currentAssistantMessage.textContent += cleanText;
    }

    // Scroll chat to bottom
    scrollChatToBottom();
  }
}

// Helper function to scroll chat to bottom
function scrollChatToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function displayTranscribedQuery(text) {
  if (!text) return;

  // Display transcribed text as a user message in the chat interface
  // Check if image should be included with transcribed audio
  const includeImageCheckbox = document.getElementById('toggleImagePrompt');
  const shouldShowImagePreview = includeImageCheckbox && includeImageCheckbox.checked && !isLlmOnlyMode();
  addChatMessage(text, true, shouldShowImagePreview);

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
socket.on('end', (data) => {
  console.log('Received end event:', data);
  receivedEndSignal = true;
  activeGeneration = false;
  pendingNewGenerationAudio = false;

  // Remove speaking indicator from current assistant message
  const assistantMessages = chatMessages.querySelectorAll('.message.assistant');
  const currentAssistantMessage = assistantMessages[assistantMessages.length - 1];
  if (currentAssistantMessage) {
    currentAssistantMessage.classList.remove('speaking');

    // Don't remove audio-playing class here - let actual audio end handle it
    // The 'end' event is for text streaming, not audio playback
  }

  // Don't hide abort button here - keep it visible during audio playback
  // hideAbortButton(); // Moved to audio completion
});
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

  const data = audioQueue.shift();
  const blob = new Blob([data], { type: 'audio/wav' });
  const arrayBuffer = await blob.arrayBuffer();

  // Check again if audio should still play after async operations
  if (!shouldPlayAudio) {
    console.log('Audio playback cancelled during processing');
    isPlaying = false;
    return;
  }

  try {
    if (currentAudioContext) {
      currentAudioContext.close();
    }

    currentAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await currentAudioContext.decodeAudioData(arrayBuffer);

    // Final check before starting playback
    if (!shouldPlayAudio) {
      console.log('Audio playback cancelled before starting');
      isPlaying = false;
      currentAudioContext.close();
      return;
    }

    const source = currentAudioContext.createBufferSource();
    const analyser = currentAudioContext.createAnalyser();
    const gainNode = currentAudioContext.createGain();

    source.buffer = audioBuffer;
    source.connect(analyser);
    analyser.connect(gainNode);
    gainNode.connect(currentAudioContext.destination);

    currentSourceNode = source;

    source.start();

    // Track First Audio timing (only for the very first audio chunk)
    if (!firstAudioStarted && userInputStartTime) {
      const firstAudioTime = (Date.now() - userInputStartTime) / 1000;
      document.getElementById('firstAudioTime').textContent = firstAudioTime.toFixed(2) + 's';
      firstAudioStarted = true;
    }

    source.onended = () => {
      isPlaying = false;
      currentSourceNode = null;

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
      if (receivedEndSignal && audioQueue.length === 0) {
        hideAbortButton();
      }

      // Only continue processing if audio should still play
      if (shouldPlayAudio) {
        setTimeout(() => processAudioQueue(), 250);
      }
    };
  } catch (err) {
    console.warn('AudioContext playback failed:', err);
    isPlaying = false;

    // Hide abort button if text streaming ended and this was the last audio
    if (receivedEndSignal && audioQueue.length === 0) {
      hideAbortButton();
    }
  }
}

async function processSystemAudioOnce(data) {
  try {
    const blob = new Blob([data], { type: 'audio/wav' });
    const arrayBuffer = await blob.arrayBuffer();

    if (currentAudioContext) {
      currentAudioContext.close();
    }
    currentAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await currentAudioContext.decodeAudioData(arrayBuffer);

    const source = currentAudioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(currentAudioContext.destination);
    source.start();
  } catch (e) {
    console.warn('System audio playback failed:', e);
  }
}

// Settings are now always visible, so initialize voice sync on load
function initializeVoiceSync() {
  try {
    const voiceSelect = document.getElementById('voiceSelect');
    const languageSelect = document.getElementById('languageSelect');
    if (voiceSelect && languageSelect && languageSelect.value === 'en') {
      fetch('/voices?lang=en')
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data && data.current) {
            const found = Array.from(voiceSelect.options).some(o => o.value === data.current);
            if (found) {
              voiceSelect.value = data.current;
              voiceSelect.dataset.prev = voiceSelect.value;
            }
          }
        })
        .catch(() => { });
    }
  } catch (e) {
    // ignore sync errors
  }
}

// Initialize RAG health check
function initializeRagHealth() {
  if (!isRagEnabled()) {
    return;
  }

  fetch("/raghealth")
    .then(res => res.json().then(data => ({ status: res.status, data })))
    .then(({ status, data }) => {
      const dbStatus = data.rag_db === "ok";
      const fpsStatus = data.rag_fps === "ok";

      // Update status message
      if (dbStatus && fpsStatus) {
        ragServerStatusText = "✅ RAG Database and RAG File Processing Server are online.";
      } else if (dbStatus && !fpsStatus) {
        ragServerStatusText = "✅ RAG Database is online. ⚠️ RAG File Processing Server is unavailable.";
      } else {
        ragServerStatusText = "❌ RAG Database is not ready yet, please wait...";
      }

      console.log(ragServerStatusText);
      const ragStatus = document.getElementById("ragStatus");
      if (ragStatus) ragStatus.textContent = ragServerStatusText;

      // Enable/disable buttons accordingly
      const importButton = document.getElementById("importRagDatabaseButton");
      const uploadButton = document.getElementById("uploadToRagButton");
      if (importButton) importButton.disabled = !dbStatus;
      if (uploadButton) uploadButton.disabled = !fpsStatus;
    })
    .catch(err => {
      ragServerStatusText = "❌ Error checking RAG server health.";
      console.error(ragServerStatusText, err);
      const ragStatus = document.getElementById("ragStatus");
      if (ragStatus) ragStatus.textContent = ragServerStatusText;

      // Disable both buttons on error
      const importButton = document.getElementById("importRagDatabaseButton");
      const uploadButton = document.getElementById("uploadToRagButton");
      if (importButton) importButton.disabled = true;
      if (uploadButton) uploadButton.disabled = true;
    });
}


// Settings panel is always visible, no close button needed

// Show language selector via URL param
window.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(window.location.search);
  if (params.get('languageselect') === '1') {
    const langRow = document.getElementById('languageRow');
    if (langRow) {
      langRow.style.display = 'block';
    }
  }

  // Initialize voice row visibility based on current language
  const voiceRow = document.getElementById('voiceRow');
  const languageSelect = document.getElementById('languageSelect');
  if (voiceRow && languageSelect) {
    voiceRow.style.display = languageSelect.value === 'en' ? 'block' : 'none';
  }

  // Fetch English voices and populate dropdown
  const voiceSelect = document.getElementById('voiceSelect');
  if (voiceSelect) {
    fetch('/voices?lang=en')
      .then(r => r.ok ? r.json() : Promise.reject(new Error('Failed to load voices')))
      .then(data => {
        if (!data || !Array.isArray(data.voices) || data.voices.length === 0) return;
        while (voiceSelect.firstChild) voiceSelect.removeChild(voiceSelect.firstChild);
        const stored = localStorage.getItem('enVoiceId');
        const current = data.current;
        let toSelect = stored || current || (data.voices[0] && data.voices[0].id);
        data.voices.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.id;
          opt.textContent = v.label || v.id;
          voiceSelect.appendChild(opt);
        });
        if (toSelect) {
          const found = Array.from(voiceSelect.options).some(o => o.value === toSelect);
          voiceSelect.value = found ? toSelect : voiceSelect.options[0].value;
        }
        if (voiceRow && languageSelect) {
          voiceRow.style.display = languageSelect.value === 'en' ? 'block' : 'none';
        }
      })
      .catch(() => { });
  }
});

// Toggle voice selector on language change
const languageSelectEl = document.getElementById('languageSelect');
if (languageSelectEl) {
  languageSelectEl.addEventListener('change', () => {
    const voiceRow = document.getElementById('voiceRow');
    if (voiceRow) {
      voiceRow.style.display = languageSelectEl.value === 'en' ? 'block' : 'none';
    }
  });
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
  // In LLM-only mode, buttons are always disabled regardless of checkbox state
  const shouldDisable = isLlmOnlyMode() || !enabled;

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
  fileInput.accept = ".pdf,.txt.,.md";
  fileInput.click();

  fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const messageBox = document.getElementById("settingsMessage");
    messageBox.textContent = "⏳ Uploading file to RAG server...";

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

// Wire English voice selection change
(function setupVoiceSelection() {
  const voiceSelectEl = document.getElementById('voiceSelect');
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

  voiceSelectEl.addEventListener('change', async () => {
    const chosen = voiceSelectEl.value;
    const label = voiceSelectEl.options[voiceSelectEl.selectedIndex]?.text || chosen;
    const voiceMsg = document.getElementById('voiceStatus');
    if (voiceMsg) { voiceMsg.textContent = `⏳ Switching voice to: ${label}...`; }

    setControlsDisabled(true);
    try {
      const res = await fetch('/voices/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang: 'en', voiceId: chosen })
      });
      if (!res.ok) throw new Error('Server rejected voice');
      const data = await res.json();
      localStorage.setItem('enVoiceId', chosen);
      voiceSelectEl.dataset.prev = chosen;
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
    }
  });
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
  document.documentElement.setAttribute('data-theme', theme);
  updateThemeIcon(theme);
  updateLogoForTheme();
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
