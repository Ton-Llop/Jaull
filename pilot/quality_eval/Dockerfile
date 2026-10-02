FROM python:3.12.12-slim-bookworm@sha256:2986c55feb36e6cae00fa1fefb454283e4b33f35e75ff8bdd123b134130be301

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/output/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 \
    && pip install --no-cache-dir --no-build-isolation --require-hashes -r /tmp/requirements.lock
ADD --checksum=sha256:010af308dae14a178fcf5e86c16fb984515cb29a7d0f0c37606faab88617f8ea https://codeload.github.com/EleutherAI/lm-evaluation-harness/tar.gz/ad8737ae7fad24cf64e50fc7fc31397bff586b9e /tmp/harness.tar.gz
RUN pip install --no-cache-dir --no-deps --no-build-isolation /tmp/harness.tar.gz \
    && rm /tmp/harness.tar.gz
WORKDIR /pilot
COPY evaluate.py suite.yaml /pilot/pilot/quality_eval/
RUN python -c "from lm_eval import simple_evaluate; from lm_eval.tasks._yaml_loader import load_yaml; from pilot.quality_eval.evaluate import validate_task; validate_task(load_yaml('/pilot/pilot/quality_eval/suite.yaml', resolve_func=True))"
ENTRYPOINT ["python", "-m", "pilot.quality_eval.evaluate"]
