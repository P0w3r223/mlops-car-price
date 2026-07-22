# The service, and nothing else: training, replay and evaluation stay outside the image.
FROM python:3.12-slim

# git: two dependencies (the modelling layer and the statistics package) are installed from
# git tags. libgomp1: LightGBM links against the OpenMP runtime, which the slim image omits -
# a missing shared object that surfaces at import time, never at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

# Serving does not need a writable filesystem or root.
RUN useradd --create-home --uid 10001 serve
USER serve

EXPOSE 8000
# The champion is resolved from the registry at startup, so the image contains no model and
# never has to be rebuilt when the model changes.
CMD ["uvicorn", "mlops_car_price.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
