FROM mambaorg/micromamba:1.5.6
ENV MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

COPY --chown=${MAMBA_USER}:${MAMBA_USER} environment.yml /tmp/environment.yml
RUN micromamba create -y -n ml-docker-api -f /tmp/environment.yml && micromamba clean --all --yes

# Usamos el entorno SÓLO en los RUN de build:
SHELL ["micromamba", "run", "-n", "ml-docker-api", "/bin/bash", "-c"]

COPY --chown=${MAMBA_USER}:${MAMBA_USER} app.py .
COPY --chown=${MAMBA_USER}:${MAMBA_USER} train_model.py .

ENTRYPOINT ["micromamba", "run", "-n", "ml-docker-api"]

ENV MODEL_DIR=/app/models
EXPOSE 5000
# ⬆️ sin ENTRYPOINT (lo controlaremos desde docker-compose)
