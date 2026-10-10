FROM python:3.12.12-slim-bookworm@sha256:2986c55feb36e6cae00fa1fefb454283e4b33f35e75ff8bdd123b134130be301

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/output/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false \
    NLTK_DATA=/usr/local/share/nltk_data
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 \
    && pip install --no-cache-dir --no-build-isolation --require-hashes -r /tmp/requirements.lock
ADD --checksum=sha256:010af308dae14a178fcf5e86c16fb984515cb29a7d0f0c37606faab88617f8ea https://codeload.github.com/EleutherAI/lm-evaluation-harness/tar.gz/ad8737ae7fad24cf64e50fc7fc31397bff586b9e /tmp/harness.tar.gz
RUN pip install --no-cache-dir --no-deps --no-build-isolation /tmp/harness.tar.gz \
    && rm /tmp/harness.tar.gz
# IFEval's checkers call nltk.download() at import when punkt_tab is missing.
# Pinned here by commit and digest so the evaluation itself never fetches it.
ADD --checksum=sha256:e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106 https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip /tmp/punkt_tab.zip
RUN python -c "import zipfile; zipfile.ZipFile('/tmp/punkt_tab.zip').extractall('/usr/local/share/nltk_data/tokenizers')" \
    && rm /tmp/punkt_tab.zip
WORKDIR /pilot
COPY evaluate.py ifeval.py suite.yaml suite_ifeval.yaml /pilot/pilot/quality_eval/
RUN python -c "from lm_eval import simple_evaluate; from lm_eval.tasks._yaml_loader import load_yaml; from pilot.quality_eval.evaluate import validate_task; validate_task(load_yaml('/pilot/pilot/quality_eval/suite.yaml', resolve_func=True))"
RUN python -c "from pilot.quality_eval import ifeval; ifeval.verify_installed_sources(); from lm_eval.tasks._yaml_loader import load_yaml; ifeval.validate_task(load_yaml('/pilot/pilot/quality_eval/suite_ifeval.yaml', resolve_func=True))"
LABEL io.jaull.quality.artifact-contract="exact-local-gguf-v1"
LABEL io.jaull.quality.suites="hellaswag-v1,ifeval-chat-v1"
ENTRYPOINT ["python", "-m", "pilot.quality_eval.evaluate"]
