FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

COPY environment.yml pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN conda env create -f environment.yml && conda clean -afy

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "qsvm-vm-compare"]
CMD ["python", "scripts/run_experiment.py", "--config", "configs/default_experiment.json"]
