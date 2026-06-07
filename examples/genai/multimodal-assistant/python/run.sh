#!/bin/bash
set -e


# Resolve script directory and run from there so relative paths are stable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment first.
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "❌ Virtual environment not found at $VENV_ACTIVATE"
    echo "   Run: bash ./install.sh"
    exit 1
fi
source "$VENV_ACTIVATE"

# Set Hugging Face cache directory
export HF_HOME=.cache/huggingface

# Resolve llima binary path early so sudo launch doesn't depend on PATH.
LLIMA_BIN="$(command -v llima || true)"

# Detect optional demo frontend location.
DEMO_APP_DIR=""
if [[ -d "$SCRIPT_DIR/simaai-genai-demo" && -f "$SCRIPT_DIR/simaai-genai-demo/app.py" ]]; then
    DEMO_APP_DIR="$SCRIPT_DIR/simaai-genai-demo"
elif [[ -f "$SCRIPT_DIR/app.py" ]]; then
    DEMO_APP_DIR="$SCRIPT_DIR"
fi
FRONTEND_AVAILABLE=false
if [[ -n "$DEMO_APP_DIR" ]]; then
    FRONTEND_AVAILABLE=true
fi

# Store PIDs for later cleanup
sudo_pid=""
demo_pid=""

# Optional arguments
app_args=()
cli_mode=false
http_only=false
run_frontend=true
run_backend=true
api_only=false
llm_only=false
ragfps_used=false
frontend_only_flag=false
backend_only_flag=false

print_help() {
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --ragfps <IP>           Connect to RAG server at given IP (port 7860 assumed)."
    echo "  --httponly              Start app server in HTTP-only mode (no Web UI features)."
    echo "  --frontend-only         Only starts app.py (UI / PiperTTS middleware)."
    echo "  --backend-only          Only starts sima_lmm (LLiMa inference)."
    echo "  --api-only              Only starts OpenAI APIs without enabling the Web UI or TTS"
    echo "  --llm-only              Disable vision features (image upload, camera) in the UI (for frontend-only mode)"
    echo "  --system-prompt-file    Provide a text file contianing custom system prompt"
    echo "  -cli                    Start the backend in CLI mode (no background process)."
    echo "  -h, --help              Show this help message and exit."
    echo ""
    echo "Note:"
    echo "  This script scans ./models, ../models, and this package root for available models."
    echo "  If multiple valid model folders are found, you will be prompted to select one."
    echo "  Frontend starts only when simaai-genai-demo/ is present."
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ragfps)
            if [[ -n "$2" ]]; then
                rag_arg="--ragserver http://$2:7860"
                ragfps_used=true
                shift 2
            else
                echo "❌ Error: --ragfps requires an IP address."
                exit 1
            fi
            ;;
        --httponly)
            http_only=true
            shift
            ;;
        --api-only)
            api_only=true
            shift
            ;;
        --llm-only)
            llm_only=true
            shift
            ;;
        --frontend-only)
            run_backend=false
            frontend_only_flag=true
            shift
            ;;
        --backend-only)
            run_frontend=false
            backend_only_flag=true
            shift
            ;;
        -cli)
            cli_mode=true
            shift
            ;;
        --system-prompt-file)
            if [[ -n "$2" && -f "$2" ]]; then
                system_prompt_file="$2"
                shift 2
            else
                echo "❌ Error: --system-prompt-file requires a valid file path."
                exit 1
            fi
            ;;            
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Disallow mutually exclusive combination that disables both
if [[ "$run_frontend" == false && "$run_backend" == false ]]; then
    echo "❌ Cannot use --frontend-only and --backend-only together."
    exit 1
fi

print_frontend_install_hint() {
    echo "💡 Frontend options require demo web app assets."
    echo "   Reinstall LLiMa and include/install simaai-genai-demo during installation."
}

if [[ "$FRONTEND_AVAILABLE" == false ]]; then
    if [[ "$frontend_only_flag" == true ]]; then
        echo "❌ --frontend-only was requested, but simaai-genai-demo was not found."
        print_frontend_install_hint
        exit 1
    fi

    if [[ "$run_frontend" == true ]]; then
        echo "ℹ️  simaai-genai-demo folder not detected. Running backend only."
        run_frontend=false
    fi

    if [[ "$ragfps_used" == true || "$http_only" == true || "$api_only" == true || "$llm_only" == true ]]; then
        echo "⚠️  Frontend-dependent arguments were provided, but frontend is unavailable."
        print_frontend_install_hint
    fi
fi

if [[ "$run_backend" == true && -z "$LLIMA_BIN" ]]; then
    echo "❌ llima command not found in .venv."
    echo "   Re-run installation to install sima_lmm wheels."
    exit 1
fi

# Trap script termination to kill background jobs
cleanup() {
    echo "🛑 Cleaning up background processes..."

    pkill -P $$ || true

    if [[ -n "$sudo_pid" ]]; then
        echo "🔪 Killing LLiMa inference server (PID: $sudo_pid)"
        sudo kill "$sudo_pid" || true
    fi

    pids=$(pgrep -f vectordb_worker.py)
    if [[ -n "$pids" ]]; then
        echo "🔪 Killing vectordb_worker.py processes: $pids"
        sudo kill $pids
    fi

    wait
}

trap cleanup SIGINT SIGTERM EXIT

wait_for_pid() {
    local pid="$1"
    local delay="${2:-1}"

    if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    while [[ -d "/proc/$pid" ]]; do
        sleep "$delay"
    done
}

# ❌ Prevent running as root
if [[ $EUID -eq 0 ]]; then
    echo "❌ Do not run this script as root or using sudo."
    echo "   Please run as a regular user."
    exit 1
fi

# Model detection logic (moved before frontend startup)
selected_model=""
selected_model_type=""

model_display_name=""
vision_model_config=""

if [[ "$run_backend" == true ]]; then
    model_search_roots=("./models/")
    valid_models=()
    model_types=()

    # Function to detect model type from vlm_config.json
    detect_model_type() {
        local config_file="$1"
        if [[ -f "$config_file" ]]; then
            # Check if vm_cfg is null or has actual configuration
            if grep -q '"vm_cfg": *null' "$config_file"; then
                echo "LLM"
            else
                echo "VLM"
            fi
        else
            echo "Unknown"
        fi
    }

    # Scan for candidate model folders across supported roots.
    for model_base_dir in "${model_search_roots[@]}"; do
        [[ -d "$model_base_dir" ]] || continue

        for dir in "$model_base_dir"*/; do
            [[ -d "$dir" ]] || continue

            foldername=$(basename "$dir")
            # Skip if it contains 'whisper-small'
            [[ "$foldername" == *"whisper-small"* ]] && continue

            # Check if required structure exists
            if compgen -G "${dir}elf_files/*.elf" > /dev/null && [[ -f "${dir}devkit/vlm_config.json" ]]; then
                valid_models+=("$dir")
                model_type=$(detect_model_type "${dir}devkit/vlm_config.json")
                model_types+=("$model_type")
            fi
        done
    done

    # Handle found models
    if [[ ${#valid_models[@]} -eq 0 ]]; then
        echo "❌ No valid models found in: ${model_search_roots[*]}"
        exit 1
    elif [[ ${#valid_models[@]} -eq 1 ]]; then
        selected_model="${valid_models[0]%/}"
        selected_model_type="${model_types[0]}"
        selected_model="$(cd "$selected_model" && pwd)"
        echo "✅ Found one model: $(basename "$selected_model") (${selected_model_type})"
    else
        echo "🔍 Multiple valid models found:"

        # Create menu options with model type
        model_options=()
        for i in "${!valid_models[@]}"; do
            model_name=$(basename "${valid_models[$i]}")
            model_type="${model_types[$i]}"
            model_options+=("$model_name ($model_type)")
        done

        select option in "${model_options[@]}"; do
            if [[ -n "$option" ]]; then
                # Find the index of the selected option
                for i in "${!model_options[@]}"; do
                    if [[ "${model_options[$i]}" == "$option" ]]; then
                        selected_model="${valid_models[$i]%/}"
                        selected_model_type="${model_types[$i]}"
                        selected_model="$(cd "$selected_model" && pwd)"
                        break
                    fi
                done
                break
            else
                echo "❌ Invalid selection."
            fi
        done
    fi
fi

# Automatically enable llm-only mode if an LLM is selected
if [[ "$selected_model_type" == "LLM" ]]; then
    echo "ℹ️ LLM detected. Automatically enabling --llm-only mode for the UI."
    llm_only=true
fi

# Extract model display name from vlm_config.json if available
if [[ -n "$selected_model" ]]; then
    config_path="${selected_model}/devkit/vlm_config.json"
    if [[ -f "$config_path" ]]; then
        vision_model_config="$config_path"
    fi
fi

# Build final args for app.py
if [[ -n "$rag_arg" ]]; then
    app_args+=($rag_arg)
fi

if [[ "$http_only" == true ]]; then
    app_args+=(--httponly)
fi

if [[ "$api_only" == true ]]; then
    app_args+=(--apionly)
fi

# Add --llm-only flag if manually specified or automatically detected
if [[ "$llm_only" == true ]]; then
    app_args+=(--llm-only)
fi

if [[ -n "$vision_model_config" ]]; then
    app_args+=(--model-config "$vision_model_config")
fi

if [[ "$cli_mode" == false && "$run_frontend" == true ]]; then
    echo "🚀 Starting demo app server in background, logs saved to app.log..."

    # Ensure log file exists before tail starts
    : > "${DEMO_APP_DIR}/app.log"

    # Start tail + colorizer *before* the app
    stdbuf -oL tail -n0 -F "${DEMO_APP_DIR}/app.log" \
      | stdbuf -oL gawk '{printf "\x1b[32m[APP] %s\x1b[0m\n", $0}' &
    demo_tail_pid=$!

    # Start the app afterwards
    (
        cd "$DEMO_APP_DIR"
        PYTHONUNBUFFERED=1 python3 -u app.py "${app_args[@]}" > app.log 2>&1
    ) &
    demo_pid=$!
fi


# Construct backend launch command
if [[ "$run_backend" == true ]]; then
    echo "🚀 Constructing LLiMa inference server start command..."
    
    llima_cmd=("$LLIMA_BIN" run "$selected_model")

    if [[ "$cli_mode" = false ]]; then
        stt_model_path="./models/whisper-small-a16w8"
        if [[ ! -d "$stt_model_path" ]]; then
            echo "❌ Whisper model path not found: $stt_model_path"
            exit 1
        fi

        llima_cmd=("$LLIMA_BIN" run "$selected_model" --mode web --stt_model_path "$stt_model_path")
    fi

    # Append system prompt file if provided
    if [[ -n "${system_prompt_file:-}" ]]; then
        llima_cmd+=(--system_prompt_file "$system_prompt_file")
    fi

    echo "${llima_cmd[@]}"
fi

if [[ "$cli_mode" = true ]]; then
    echo "🖥️  Starting LLiMa in CLI mode..."
       
    echo -n "📸 Would you like to provide an image for the model? (y/n): "
    read -r want_image
    
    if [[ "$want_image" == "y" || "$want_image" == "Y" ]]; then
        echo -n "📂 Please give the image path [./images/default_image.jpg] (press Enter for default): "
        read -r image_path
        
        # Use default if user pressed enter without typing anything
        if [[ -z "$image_path" ]]; then
            image_path="./images/default_image.jpg"
        fi
        
        # Check if the image file exists
        if [[ -f "$image_path" ]]; then
            echo "✅ Using image: $image_path"
            (echo "set image $image_path"; cat) | "${llima_cmd[@]}"
        else
            echo "❌ Image file not found: $image_path"
            echo "📝 Continuing without image..."
            "${llima_cmd[@]}"
        fi
    else
        "${llima_cmd[@]}"
    fi
    
    exit 0
fi

if [[ "$run_backend" == true ]]; then
    echo "🚀 Starting LLiMa inference server in background, check logs in console.log..."

    : > console.log

    # 1️⃣ Build escaped command string
    backend_cmd_str=""
    for arg in "${llima_cmd[@]}"; do
        backend_cmd_str+=$(printf "%q " "$arg")
    done

    # 2️⃣ Use a PID file to track spawned backend
    PID_FILE=".llima.pid"
    rm -f "$PID_FILE"

    # 3️⃣ Ensure sudo auth and launch backend detached from this shell.
    if [[ -n "${SUDO_PASSWORD:-}" ]]; then
        echo "🔐 Using provided SUDO_PASSWORD" >> console.log
        # We use tee for both the log and the PID to ensure the 'sima' user is the writer
        bash -c "echo \"$SUDO_PASSWORD\" | sudo -S -E bash -c '$backend_cmd_str 2>&1' | tee -a console.log & echo \$! | tee $PID_FILE > /dev/null" </dev/null >/dev/null 2>&1 &
        sleep 0.2
    elif sudo -n true 2>/dev/null; then
        echo "✅ Using passwordless sudo" >> console.log
        sudo -E bash -c "$backend_cmd_str 2>&1" | tee -a console.log &
        echo $! | tee "$PID_FILE" > /dev/null
    else
        echo "🔔 No SUDO_PASSWORD env set — falling back to interactive sudo." >> console.log
        
        # Refresh sudo credentials interactively before backgrounding
        if ! sudo -v; then
             echo "❌ Sudo authentication failed." | tee -a console.log
             exit 1
        fi

        sudo -E bash -c "$backend_cmd_str 2>&1" | tee -a console.log &
        echo $! | tee "$PID_FILE" > /dev/null
    fi

    # 6️⃣ Read back the PID and tail the logs
    if [[ -f "$PID_FILE" ]]; then
        sudo_pid="$(cat "$PID_FILE")"
        echo "🔎 Backend PID: $sudo_pid" | tee -a console.log
    else
        echo "⚠️ Backend may not have started properly (PID file missing)." | tee -a console.log
    fi
    
    # 7️⃣ Stream logs live
    tail -n 0 -F console.log | awk '{ print "\033[0;33m[LLiMa] " $0 "\033[0m" }' &
fi


# Wait until processes exit
if [[ "$cli_mode" == false ]]; then
    if [[ "$run_frontend" == true && -n "$demo_pid" ]]; then
        wait $demo_pid
    fi
    if [[ "$run_backend" == true && -n "$sudo_pid" ]]; then
        wait_for_pid "$sudo_pid"
    fi
else
    if [[ "$run_backend" == true && -n "$sudo_pid" ]]; then
        wait_for_pid "$sudo_pid"
    fi
fi
