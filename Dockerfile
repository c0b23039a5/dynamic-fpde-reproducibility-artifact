FROM python:3.14-slim
WORKDIR /artifact
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /artifact/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /artifact
CMD ["python", "scripts/build_reproducibility_tables.py", "--input", "results_precomputed", "--output", "generated", "--expected-tasks", "72", "--expected-seeds", "10"]
