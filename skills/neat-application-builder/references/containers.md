# Containers

Start from the Apps example `examples/docker/parallel-apps`. Its README section
"Container Settings" explains each `docker run` setting. Keep that shape: the
container runs the DevKit's Neat, so build the binary against the same Core.

Beyond the README:

- Run Core 0.6.0 or later. Earlier Cores cannot identify the board inside a
  container, and containers sharing an extraction directory can overwrite each
  other's unpacked models.
- Each container unpacks its model archive at startup, into `/tmp` or
  `SIMA_MPK_EXTRACT_ROOT`. The unpacked model is larger than the archive, and on
  a tmpfs `/tmp` it occupies RAM per container.
- All containers share one pool of device memory. Treat
  `resource.device_memory_exhausted` as a signal to reduce streams, resolution,
  or buffering.
- Stop the containers before installing or updating Neat on the DevKit. The
  installer's EV74 firmware activation refuses to run while another Neat
  process holds the accelerator.

Building and deploying images with the SDK's `dk container` commands belongs to
the Neat Development Environment documentation.
