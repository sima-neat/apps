# Image of one Apps example, built by parallel-apps.sh from the installed Apps
# bundle or an Apps checkout. BINARY is the example's executable, relative to
# that root.
#
# The image holds the example's binary, its src/common files and config, and its
# model. The DevKit's installed Neat and system libraries are mounted read-only
# when the container starts, so the container uses the same Neat as a normal run
# on the DevKit.
FROM scratch

ARG EXAMPLE
ARG BINARY
ARG CONTAINER

# GStreamer keeps its plugin cache under HOME; /tmp is the container's
# writable tmpfs.
ENV PATH=/usr/bin:/bin HOME=/tmp

# The working directory mirrors the Apps root, so the relative model and label
# paths in the example's config resolve as they do there.
WORKDIR /apps
COPY ${BINARY} /app
COPY examples/${EXAMPLE}/src/common/ examples/${EXAMPLE}/src/common/
COPY models/${CONTAINER}/ models/

# ENTRYPOINT cannot expand build arguments, so the config also goes to a
# fixed path.
COPY examples/${EXAMPLE}/src/common/config.yaml config.yaml
ENTRYPOINT ["/app", "--config", "config.yaml"]
