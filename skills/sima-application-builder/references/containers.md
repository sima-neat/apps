# Containers

Start from the Apps example `docker/parallel-apps`. Its README section "Container Settings"
explains each `docker run` setting. The container runs the DevKit's Neat, so build the binary
against the same Core.

Beyond that README:

- Run Core 0.6.0 or later. Earlier Cores cannot identify the board inside a container, and
  containers that share an extraction directory can overwrite each other's unpacked models.
- Each container unpacks its model archive at startup, into `SIMA_MPK_EXTRACT_ROOT` when it is
  set, and the load fails when that folder is not writable. Otherwise it uses the first writable
  of a mounted NVMe, `/data/simaai/coprocessing/models/`, `$TMPDIR/simaai/coprocessing/models`
  (`/tmp` when `TMPDIR` is unset), and `./tmp/model_extract`. The unpacked model is
  larger than the archive, and on a tmpfs `/tmp` it takes RAM in every container.
- All containers share one pool of device memory. `resource.device_memory_exhausted` means the
  containers together need fewer streams, a lower resolution, or less buffering.
- Stop the containers before installing or updating Neat on the DevKit. The installer's EV74
  firmware activation refuses to run while another Neat process holds the accelerator.

Building and deploying images with the SDK's `dk container` commands belongs to the Palette SDK
documentation.
