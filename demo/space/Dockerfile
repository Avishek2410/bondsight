# Hugging Face Space (Docker SDK): the inspector in demo (visitor) mode on port 7860.
# Built by tools/build_space.py into out/space/; nothing here is stored between visitors.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the app as user 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /app
COPY synth/ synth/
COPY inspector/ inspector/
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --no-cache-dir ./synth ./inspector
COPY models/ models/
COPY samples/ samples/
COPY config.yaml CREDITS.md LICENSE ./
USER user

EXPOSE 7860
CMD ["wbinspect", "--config", "/app/config.yaml", "--host", "0.0.0.0"]
