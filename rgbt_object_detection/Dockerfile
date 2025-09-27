FROM nvcr.io/nvidia/pytorch:23.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV CRASH_HANDLER=FALSE
ARG USERID

RUN apt-get update && apt-get install -y byobu libgl1-mesa-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' --uid $USERID dockuser

USER dockuser

RUN pip --no-cache-dir install Cython
COPY ./requirements.txt /workspace
RUN pip install --upgrade pip && pip --no-cache-dir install -r /workspace/requirements.txt

WORKDIR /workspace

CMD ["python", "benchmarking.py"]
