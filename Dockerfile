# ROCm/gfx1201 tensor-parallel container build for ds4.
#
# This image MUST be built on the target host, not pulled pre-built from a registry.
# Two host-specific properties get baked into the binaries at compile time:
#   - CPU codegen: the CFLAGS use -march=native, so the resulting binaries carry
#     instructions selected for whatever CPU `docker build` runs on. A binary built
#     on a different CPU generation (e.g. an image built in CI and shipped here) can
#     fail immediately with an illegal instruction the moment it hits a code path
#     that used an instruction this machine's CPU doesn't have. That has already
#     happened once on this project.
#   - GPU codegen: --offload-arch is pinned to gfx1201 (AMD Radeon AI PRO R9700) via
#     ROCM_ARCH below, matching the cards physically installed in this host.
# `docker build` / `docker compose build` executes RUN steps on the daemon's host
# CPU, so building locally (never `docker pull`-ing a prebuilt image for this
# Dockerfile) is what keeps codegen matched to the machine actually running it.
#
# The ROCm version is pinned to 7.2.3 via repo.radeon.com's versioned apt path,
# matching this host's own `/opt/rocm-7.2.3` install exactly (see `apt-cache policy
# rocm-hip-sdk` on the host). This isn't cosmetic: the newer 7.2.4 RHEL package set
# was tried first and fails to compile the ROCm sources at all (a real upstream bug
# in that package's HIP headers -- `__AMDGCN_WAVEFRONT_SIZE` is referenced but never
# defined for gfx1201). Pinning to the exact version already proven to work on this
# machine avoids re-discovering that class of problem.

ARG BASE_IMAGE=ubuntu:24.04
ARG ROCM_APT_VERSION=7.2.3
ARG ROCM_ARCH=gfx1201

FROM ${BASE_IMAGE} AS rocm-base
ARG ROCM_APT_VERSION

RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_APT_VERSION} noble main" \
        > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 1001\n' \
        > /etc/apt/preferences.d/rocm-pin.pref \
    && apt-get update -qq

FROM rocm-base AS builder
ARG ROCM_ARCH

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        gcc g++ make git libxml2 \
        hip-dev hipblas-dev hipblaslt-dev hipcub-dev rocprim-dev rocm-device-libs rocwmma-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN make -j"$(nproc)" rocm ROCM_ARCH=${ROCM_ARCH}

FROM rocm-base AS runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        hip-runtime-amd hipblas hipblaslt curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/ds4 /src/ds4-server /src/ds4-bench /src/ds4-eval /src/ds4-agent /usr/local/bin/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=1200s --retries=3 \
    CMD curl -sf http://localhost:8000/v1/models -o /dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/ds4-server"]
