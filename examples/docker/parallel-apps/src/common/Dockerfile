# Image of one installed Apps example. The build context is prebuilt-apps/.
#
# The image holds only the example's binary and runtime files. The board's
# installed Neat and system libraries are mounted read-only when the container
# starts (see compose.yaml), so the container always uses the same Neat as a
# normal run on the board.
FROM scratch

ARG EXAMPLE
ARG BINARY

# GStreamer keeps its plugin cache under HOME; /tmp is the container's
# writable tmpfs.
ENV PATH=/usr/bin:/bin HOME=/tmp

# The working directory mirrors prebuilt-apps/, so the relative model and label
# paths in the example's config resolve as they do on the board.
WORKDIR /apps
COPY examples/${EXAMPLE}/src/common/ examples/${EXAMPLE}/src/common/
COPY examples/${EXAMPLE}/src/cpp/pre-built/${BINARY} /app

# compose.yaml mounts the example's own config.yaml here.
ENTRYPOINT ["/app", "--config", "/apps/config.yaml"]
