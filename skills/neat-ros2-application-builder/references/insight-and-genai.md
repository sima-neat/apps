# Insight and GenAI integration

## Useful results before visualization

Publish numerical/typed results independently of optional rendering. A result
recorder is not automatically an Insight visualization sink. State whether the
inference node renders matched previews, an output node joins images/results, or
an integrated Neat graph sends video/metadata directly.

For Neat example live visualization, use Insight through the supported public
video/metadata APIs. Keep ordinary ROS outputs available for downstream consumers.
Offline saved outputs are useful for numerical and visual validation. Respect an
existing application's visualization contract rather than adding another UI.

For overlays, use the exact source frame and correct coordinate transform. Bound
frame/result caches and record expired/missing pairs. Rendering must not mutate
inference input. If an already-annotated image is sent as video, do not duplicate
its annotations as metadata without an intended rendering contract.

Expose enable flags, Insight host/ports, channel mapping, output FPS and optional
saving in configuration. Distinguish inference rate from rendering and sender
caps. Avoid encoding/rendering unused previews where the graph permits it.
Multiple nodes/apps must not accidentally send different images to the same
channel. Allocate coherent video/metadata port ranges and stable stream IDs;
metadata timestamps must match the video PTS contract.

Validate beyond sender startup: inspect output messages, confirm receiver traffic,
and check that Insight's rendered video advances with the expected overlay/depth
geometry. A send counter does not prove successful browser playback. Retained
stills and paused channels are not live FPS. Use a bounded capture or screenshot
when useful, and state if the receiving UI could not be observed.

## VLM request contract

An isolated VLM node can subscribe to images and accept prompts independently of
continuous vision nodes. Image callbacks retain only a bounded latest-frame
reference; initialization and HTTP/model work run outside the executor callback.

Make modes explicit:

| Mode | Frame selection | Prompt |
| --- | --- | --- |
| On-demand full frame | Snapshot the latest sufficiently recent image when accepting the request | Supplied by the user |
| Periodic region/crop | A documented detection selection rule at a bounded interval | Configured prompt |

These are useful defaults for an illustrative app, not a reason to override an
existing user-selected ROI contract. If crops are used, document class filtering,
selection/ranking, clipping, minimum size, stale-frame handling and behavior when
there is no eligible detection. Publish the selected crop/ROI and source identity
so the user can see exactly what the model received.

Keep request ID, prompt and accepted image together until completion; new images
must not replace the snapshot mid-request. Account for frame age using consistent
clock semantics. For bag replay, compare local reception age when that is the
contract rather than subtracting an old recording stamp from wall time.

For streamed answers, publish bounded text chunks with request ID and elapsed
time since acceptance, followed by a final answer or explicit error. Preserve
ordering and distinguish first-chunk latency from total latency. Chunk topics
normally carry current events; retained final answer/status can help late
subscribers. Document QoS and how to echo long responses without CLI truncation.
Do not interpret an ellipsis printed by the CLI as missing backend output.

Retain a small bounded queue or reject excess requests with a clear busy result.
Expose timeout and token limits. Report model loading and ready states. Test a
request while other models continue, if concurrency is advertised. Sharing the
accelerator can increase response latency even with correct asynchronous code.
Do not promise standalone latency during concurrent execution.

## Service setup and HTTPS

Check the installed public GenAI APIs and the selected backend's supported
endpoints/stream format. Use the matching prepared model directory or documented
service. Avoid silently starting a second service on an occupied port. A service
accepting TCP connections may still be loading the model.

For HTTPS, use a verified client context. Trust the backend's generated or
configured certificate/CA when the documented local workflow uses a self-signed
certificate. Keep hostname verification enabled and ensure the endpoint matches
the certificate's SAN. Do not fix trust failures with `verify=False` or an
unverified context. Resolve certificate paths using the documented config/asset
layout, including installed packages; a certificate may be generated after node
startup, so validate it at the appropriate connection/readiness boundary.

Use explicit modern TLS bounds in controlled HTTPS test servers, such as
`context.minimum_version = ssl.TLSVersion.TLSv1_2`, as recommended by
[CodeQL's TLS guidance](https://codeql.github.com/codeql-query-help/python/py-insecure-protocol/).
Test successful default HTTPS, explicit CA, untrusted CA, wrong hostname and
missing certificate; an HTTP-only fixture cannot validate HTTPS trust behavior.
Keep certificates/keys ephemeral test assets, and never ship credentials or test
private keys as example configuration.
