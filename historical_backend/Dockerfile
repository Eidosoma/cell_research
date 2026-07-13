FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /opt/legacy-adapter
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock \
    && python -m pip check

COPY legacy_adapter.py validate_backend.py ./

# The image intentionally has no historical source. A non-root runtime receives
# the S02 quarantine as a read-only bind mount at /historical-src.
USER 65532:65532
ENTRYPOINT ["python", "/opt/legacy-adapter/legacy_adapter.py"]
