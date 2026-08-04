# Reference submission container for EchoRisk-MICCAI 2026.
#
# TASK_ID is fixed at build time, not at run time (the container is
# invoked with no arguments — see docs/HARNESS_SPEC.md §5). Build one
# image per task you are submitting to.
#
#   docker build --build-arg TASK_ID=1 -t <team>-task1:final .
#   docker build --build-arg TASK_ID=2 -t <team>-task2:final .
#   docker build --build-arg TASK_ID=3 -t <team>-task3:final .

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Replace with your actual dependencies. This reference image only needs
# Python + a CSV standard-library round trip, so no ML framework is
# installed here — add torch/tensorflow/etc. as your real model requires.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG TASK_ID
ENV TASK_ID=${TASK_ID}

COPY infer.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy your model weights and additional code here, e.g.:
# COPY model/ ./model/

ENTRYPOINT ["./entrypoint.sh"]
