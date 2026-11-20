# Neat GenAI Studio — ROS 2

This is an independent ROS 2 package, `neat_genai_studio`. Build and run it from
this directory. Nodes, launch, configuration, labels and implementation helpers
are local; no `neat_apps` package or original example source is imported.

## Structure and node roles

```text
neat-genai-studio/
├── package.xml
├── CMakeLists.txt
├── launch/app.launch.py
└── src/
    ├── common/config.yaml
    └── python/
        ├── studio_node.py
        ├── studio_tls.py
        └── image_transport.py
```

[studio_node.py](src/python/studio_node.py) declares its ROS interfaces and execution flow directly.
[studio_tls.py](src/python/studio_tls.py) selects the trusted certificate for HTTPS requests.
`main()` declares image/request subscriptions and response publication.
`receive()` admits bounded requests; `perform()` makes the HTTP inference call
on a worker. The local `server/` and `api/` directories contain the unchanged
model service and backend; `run.sh` starts them. No image source is started.

## Prepare and configure

Install ROS 2 with:

```bash
sima-cli neat install ros2@v3.0.0
```

Use the matching Neat Python environment and installed colcon build tools.
Use the [native model preparation guide](../../../genai/neat-genai-studio/README.md#prepare-the-model)
for the same model artifacts and precision. Edit [src/common/config.yaml](src/common/config.yaml).
Model and dataset files are user-provided; use absolute paths, or paths relative
to this example directory. Shipped labels are under `src/common/`.

| Configuration | Meaning |
| --- | --- |
| `server` | Model catalog, loaded models and model-server endpoints |
| `app` | Local backend endpoints, web service and speech engines |
| `ros.studio_url` | Existing backend URL; defaults to the configured local web service |
| `ros.tls_ca_file` | Optional trusted PEM CA/certificate; relative paths resolve beside the YAML file |
| `ros.request_timeout_seconds` | Maximum backend request time, default 120 seconds |
| `ros.pending_requests` | Bound on pending/active requests, default 1; excess requests get an error |
| `ros.image_topic` | Latest optional image for image-bearing chat requests, default `image` |

These are startup YAML settings, not dynamic ROS parameters. Restart after edits.

## Build on Modalix

From this example directory:

```bash
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
colcon build
source install/setup.bash
```

Only this package is built. `build/`, `install/` and `log/` stay in this folder.
Rebuild after Python or launch edits. In every new terminal source ROS, activate
Neat and source this installation. Configuration edits only require a restart
when using the source config below.

## Run

Prepare the backend dependencies with this folder’s `setup.sh`, as described in
the native preparation guide. Start the local backend in another terminal with:

```bash
CONFIG_PATH="$PWD/src/common/config.yaml" ./run.sh
```

The default local URL uses HTTPS. The ROS client trusts the backend's generated
`.local-certs/server.crt` in this app's checkout (or installed package when using
the installed backend/config). If the backend uses `NEAT_TLS_CERT` and
`NEAT_TLS_KEY`, export the same `NEAT_TLS_CERT` in the ROS terminal or set
`ros.tls_ca_file` to its certificate/issuing CA. The ROS client never needs the
private key. Start the backend before sending requests so its certificate exists.

Alternatively use an existing backend and set `ros.studio_url`. An explicit URL
uses the system trust store; for a self-signed or privately issued certificate,
set `ros.tls_ca_file` to a trusted local PEM file. For example:

```yaml
ros:
  studio_url: https://studio.example:5000
  tls_ca_file: /path/to/studio-ca.pem
  pending_requests: 1
```

Relative certificate paths resolve beside the YAML file. When an installed ROS
node uses a config stored outside the backend's app directory, set `tls_ca_file`
explicitly if the backend runs from a different checkout/installation. Certificate
and hostname verification remain enabled; the URL hostname/IP must appear in the
certificate. HTTP (`app.web.https: false` or an explicit HTTP URL) needs no TLS
settings. These settings apply to chat, speech, transcription and translation.
The ROS node uses the same model service, generation settings and speech engines.

```bash
ros2 launch neat_genai_studio app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

`namespace:=demo` changes the namespace. Ctrl+C stops the launched nodes.
A component failure stops the launch. Omit `config` to use the installed config copy.

## Results and Insight

Publish JSON requests on `/neat/requests`; responses arrive on `/neat/responses`.
An optional `/neat/image` subscription supplies BGR/RGB images when a request
sets `use_image: true`. Operations are `chat`, `speech`, `transcribe`, `translate`.

```bash
ros2 topic echo /neat/responses
# In a second sourced terminal, choose a model loaded in your backend:
ros2 topic pub --once /neat/requests std_msgs/msg/String \
  '{data: "{\"id\":\"demo\",\"operation\":\"chat\",\"body\":{\"model\":\"<loaded-model-name>\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":false}}"}'
```

The backend is also available in Insight’s GenAI tab. ROS responses are not
a video stream; configure that backend URL in Insight. Audio responses include
base64 audio and timing metadata. The original output-quality requirement
applies; speech throughput is recorded separately.

## Extend and validate

Edit [studio_node.py](src/python/studio_node.py) for model behavior and
[launch/app.launch.py](launch/app.launch.py) for wiring. Add executable scripts
to this folder’s `CMakeLists.txt` and dependencies to `package.xml`.

The migration preserves model routes, preprocessing, decoding, queue bounds and
worker topology. Build/import checks alone do not establish accuracy or FPS
parity. See the [validation status](../../README.md#validation-status) for measured
coverage and remaining runtime/hardware limitations. Physical MIPI/USB and
Docker execution remain TODO where applicable.

See the [node development guide](../../NODE_DEVELOPMENT.md) for creating
your own app, and the [ROS overview](../../../../ROS2.md) for validation limits.
