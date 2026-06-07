# GenAI Demo App Server

The demo app server is a middleware layer between the User Interface and the LLM backend.

# To install dependencies

This project implements TTS through PiperTTS, and will pull down models during installation. Please make sure you have Internet connection on the DevKit when you run the `install.sh` script. After installation and first round of test, the DevKit can work in offline mode without connecting to the Internet. 

```
./install.sh
```

The `install.sh` script will setup the virtual environment under `.venv`. 


# To run the app
```
./run.sh --help

Usage: ./run.sh [options]

Options:
  --ragfps <IP>       Connect to RAG server at given IP (port 7860 assumed).
  --httponly          Start app server in HTTP-only mode (can't capture audio or video feature, this is a compatibility mode in case client only supports http).
  --sample-audio val  Enable sample filler audio (true|false). Default: false
  --frontend-only     Only starts app.py (UI / PiperTTS middleware).
  --backend-only      Only starts sima_utils (LLiMa inference).
  --api-only          Only starts OpenAI APIs without enabling the Web UI or TTS.
  --use-sima-lmm      Use the new sima-lmm module instead of sima-utils.
  -cli                Start the backend in CLI mode (no background process).
  -h, --help          Show this help message and exit.

Note:
  This script scans the parent directory for available models.
  If multiple valid model folders are found, you will be prompted to select one.
```


# To package for deployment

The included metadata.json file is an example configuration that defines the model package and streamlines the installation process.
If new models are tested against sima-utils, you can create a new metadata.json modeled after the sample.

To build a deployable archive:
```
./build-dist.sh
```
This will generate a file called:

```
simaai-genai-demo.tar.gz
```

Once the package is installed, to install, use:

```
sima-cli -m http://link-to-your/metadata.json
```
