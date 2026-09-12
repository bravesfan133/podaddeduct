FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    vainfo \
    intel-media-va-driver \
    libva2 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt
COPY podaddeduct ./podaddeduct
COPY scripts ./scripts
COPY README.md .

ENV DATA_DIR=/data
ENV LIBVA_DRIVER_NAME=iHD
VOLUME ["/data"]
EXPOSE 7887
CMD ["python", "-m", "podaddeduct"]
