FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TESSDATA_PREFIX=/usr/share/tessdata \
    OMP_THREAD_LIMIT=1 \
    PATH=/opt/venv/bin:$PATH

# Python 3.12, Tesseract 5 (LSTM) and the fonts used both to measure glyphs and
# shipped to Gotenberg when regenerating OCR'd pages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv tesseract-ocr curl ca-certificates libglib2.0-0t64 \
        fonts-noto-core fonts-noto-mono fonts-hosny-amiri fonts-liberation2 \
    && rm -rf /var/lib/apt/lists/*

# Most accurate LSTM models ("tessdata_best"); override with --build-arg OCR_LANGS="ara eng fra deu"
ARG OCR_LANGS="ara eng fra"
ARG TESSDATA_REPO=https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main
RUN mkdir -p ${TESSDATA_PREFIX} \
    && cp -r /usr/share/tesseract-ocr/*/tessdata/configs /usr/share/tesseract-ocr/*/tessdata/tessconfigs ${TESSDATA_PREFIX}/ \
    && cp /usr/share/tesseract-ocr/*/tessdata/osd.traineddata ${TESSDATA_PREFIX}/ \
    && for l in ${OCR_LANGS}; do \
         curl -fsSL --retry 5 -o ${TESSDATA_PREFIX}/$l.traineddata ${TESSDATA_REPO}/$l.traineddata || exit 1; \
       done \
    && tesseract --list-langs

WORKDIR /srv
COPY requirements.txt .
RUN python3 -m venv /opt/venv && pip install -r requirements.txt
COPY app ./app

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
