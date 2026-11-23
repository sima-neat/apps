# Package and node design

## Make one example understandable in one directory

For a standalone example, a useful layout is:

```text
my_perception_app/
  package.xml
  CMakeLists.txt                  # Or ament_python setup files
  README.md
  launch/app.launch.py
  src/common/config.yaml
  src/common/labels.txt           # If the model needs labels
  src/python/input_node.py
  src/python/detection_node.py
  src/python/output_node.py      # If a separate output role is useful
  src/python/image_transport.py
  tests/
```

This is an example, not a required layout for existing packages. C++ nodes can
live under `src/cpp`. Keep application-specific helpers local; a single example
should not depend on importing another example's private source or building a
large shared dispatcher. Reuse a stable installed utility library when one is
actually part of the distribution. Avoid duplicating an inference runtime under
a generic `ros_node.py` that hides the model flow.

Put publishers, subscriptions, declared parameters and worker startup where a
reader can find them in the node constructor. Use explicit methods for model
initialization, input preparation, submission, completion, decoding and close.
Keep substantive graph/model setup visible; helpers should have a concrete role
such as packing image planes or decoding a specific model output.

## Choose source and inference boundaries

Typical reusable composition:

```text
input node -> image topic -> inference node -> result topic
                         \-> another model node
frames + correlated results -> optional output node -> Insight / saved output
```

The input node owns file reading, camera capture or video decoding. The inference
node accepts the documented image contract and owns Neat model initialization
and inference. Do not decode an already decoded image again in every model node.
The output node owns visualization/recording, not model execution.

For a high-throughput live-camera mode, decoding and inference may share one
Neat graph inside a ROS node, publishing results without transmitting every raw
frame through DDS. Keep a separate image-topic mode when external ROS cameras
are required. Document the different image-topic availability and performance
boundaries. Preserve the requested node decomposition unless a measured reason
supports proposing a topology change.

Provide explicit launch modes where relevant:

| Mode | Meaning |
| --- | --- |
| App-owned source | Launch the configured input node and the inference node(s) |
| External source | Start subscribers and wait for the configured image topic |
| Combined live graph | The inference node owns capture/decode in its Neat graph |

Names such as `source:=none` or `source:=live_graph` are application choices,
not ROS standards. Expose only implemented modes. A file source decoding to BGR
cannot feed an NV12-only node without a documented, tested conversion/model path.
Trace source configuration through publication and subscription when reviewing it.

## Contracts that belong in the README

Include a node table: role, included work, subscriptions, publications and source
file. Include a topic table with resolved name/remapping, message type, encoding,
units/coordinate frame, header identity, QoS and publication condition.

Prefer standard ROS messages when they fit. Dense depth/masks belong in typed
image/tensor messages, not huge JSON float arrays. For JSON result topics, document
the schema and distinguish an empty result from an error. Custom messages are
appropriate for an established integration contract, not automatically required
for every AI example. Separate incompatible generated interfaces from helper
modules if they would otherwise collide in one Python package namespace.

Document every user-facing parameter: type, default, accepted range, meaning,
startup-only versus mutable, and consequences of a change. Label plain YAML
settings accurately: loading YAML directly does not make them dynamic ROS
parameters. State relative-path rules and use one explicit launch override policy
so defaults do not silently overwrite YAML values.

A complete run path includes:

1. Required ROS/Neat environment, supported models and acquisition instructions.
2. Build/source commands from this example directory.
3. Required model/source/Insight config edits.
4. Launch command for app-owned and external sources, where supported.
5. Commands to inspect results and change documented controls.
6. Expected meaningful output, shutdown and known validation limits.

Model-free readiness, model loading and inference readiness are different states.
A launched process is not proof that a model or source is usable. Inference should
remain off callbacks during initialization too; publish useful readiness/error
state when startup can take time.
