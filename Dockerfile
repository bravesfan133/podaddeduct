FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt
COPY podaddeduct ./podaddeduct
COPY scripts ./scripts
COPY README.md .
# Pre-download the whisper model so first transcription isn't a surprise.
# STT_MODEL build arg matches compose default (tiny/base/small).
ARG STT_MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('$STT_MODEL', device='cpu', compute_type='int8')"

ENV DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8080
CMD ["python", "-m", "podaddeduct"]