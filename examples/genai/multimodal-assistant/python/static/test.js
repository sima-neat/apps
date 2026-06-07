// Variables to hold media recorder and audio data
let mediaRecorder;
let audioChunks = [];

// Function to handle recording
async function startRecording() {
    // Access the microphone
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Initialize the MediaRecorder
    mediaRecorder = new MediaRecorder(stream);

    // Event handler for when audio data becomes available
    mediaRecorder.ondataavailable = function(event) {
        audioChunks.push(event.data);  // Collect audio data chunks
    };

    // When recording stops, send the audio to the server
    mediaRecorder.onstop = async function() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });  // Convert chunks to audio blob
        audioChunks = [];  // Reset the audio chunks

        // Send audio to server
        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.wav');

        try {
            const response = await fetch('http://127.0.0.1:8080/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const jsonResponse = await response.json();
                const responseText = jsonResponse.processed_text;  // Get the response text
                playTextAsSpeech(responseText);  // Play the response text as audio
            } else {
                console.error('Server error:', response.status);
            }
        } catch (error) {
            console.error('Fetch error:', error);
        }
    };

    // Start recording
    mediaRecorder.start();
    console.log("Recording started...");
}

// Function to stop recording
function stopRecording() {
    mediaRecorder.stop();
    console.log("Recording stopped...");
}

// Function to play text as speech
function playTextAsSpeech(text) {
    if ('speechSynthesis' in window) {
        const speech = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(speech);
    } else {
        console.error("Speech synthesis not supported");
    }
}

// Attach event listeners to buttons
document.getElementById("startRecord").addEventListener("click", function() {
    startRecording();
    document.getElementById("startRecord").disabled = true;
    document.getElementById("stopRecord").disabled = false;
});

document.getElementById("stopRecord").addEventListener("click", function() {
    stopRecording();
    document.getElementById("startRecord").disabled = false;
    document.getElementById("stopRecord").disabled = true;
});

