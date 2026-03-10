---
inclusion: always
---

# TrustMem Local Setup

When the user wants to set up TrustMem, **do NOT jump straight into installation**. First ask key questions to determine the right path.

## Decision Flow

### Question 1: Which AI tool?
Ask: "You're using Kiro, Cursor, or Claude Code? (or multiple?)"
This determines which config files to generate.

### Question 2: MatrixOne database
Ask: "Do you already have a MatrixOne database running? If not, I can help you set one up. You have two options:
1. **Local Docker** (recommended for development) — I'll start one for you with docker-compose
2. **MatrixOne Cloud** (free tier available) — register at https://cloud.matrixorigin.cn, no Docker needed"

Based on the answer:
- **Already have one** → ask for the connection URL (host, port, user, password, database)
- **Local Docker** → follow Docker setup below
- **MatrixOne Cloud** → guide user to register, then get connection URL from console

### Question 3: Embedding provider
Ask: "For memory search quality, TrustMem needs an embedding model. Options:
1. **Existing service** (recommended if available) — OpenAI, Ollama, or any embedding endpoint you already run. No download, no cold-start.
2. **Local** (default, free, private) — downloads ~80MB model on first install, first query takes a few seconds to load into memory. Subsequent queries are fast.
3. **OpenAI** — better quality, needs API key, no cold-start delay."

## Execution Paths

### Path A: Local Docker + Local Embedding (most common)

```bash
# 1. Start MatrixOne
docker compose up -d    # if in TrustMem repo
# or:
docker run -d --name matrixone -p 6001:6001 -v ./data/matrixone:/mo-data --memory=2g matrixorigin/matrixone:latest

# 2. Wait for healthy (~30-60s on first start)
docker ps --filter name=matrixone

# 3. Virtual environment (if not already in one)
python3 -m venv .venv && source .venv/bin/activate

# 4. Install
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'trust-mem-lite[local-embedding]'

# 5. Configure (in user's project directory)
cd <user-project>
trustmem init
```

### Path B: MatrixOne Cloud

```bash
# 1. User registers at https://cloud.matrixorigin.cn (free tier)
# 2. Get connection info from cloud console: host, port, user, password

# 3. Virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 4. Install
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'trust-mem-lite[local-embedding]'

# 5. Configure with cloud URL
cd <user-project>
trustmem init --db-url 'mysql+pymysql://<user>:<password>@<host>:<port>/<database>'
```

### Path C: Existing MatrixOne

```bash
# 1. Virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'trust-mem-lite[local-embedding]'

# 3. Configure with existing DB
cd <user-project>
trustmem init --db-url 'mysql+pymysql://<user>:<password>@<host>:<port>/<database>'
```

### Embedding provider flags (for any path)

```bash
# Local (default) — no extra flags needed
trustmem init

# OpenAI
trustmem init --embedding-provider openai --embedding-api-key sk-...

# Existing service (Ollama, custom endpoint, etc.)
trustmem init --embedding-provider openai --embedding-base-url http://localhost:11434/v1
```

## After any path

```bash
# Verify
trustmem status

# Tell user to restart their AI tool
```

## Troubleshooting
- MatrixOne won't start → `docker logs trustmem-matrixone` to check errors
- Port 6001 in use → edit `.env` to change `MO_PORT`, then `docker compose up -d`
- Can't connect to DB → MatrixOne needs 30-60s on first start, wait and retry
- Cloud connection refused → check firewall/whitelist settings in cloud console
- **Docker permission denied** → `sudo usermod -aG docker $USER && newgrp docker`
- **Image pull slow/timeout** → configure Docker mirror in `/etc/docker/daemon.json`, add `"registry-mirrors": ["https://docker.1ms.run"]`, then `sudo systemctl restart docker`
- **Docker not installed** → suggest MatrixOne Cloud (https://cloud.matrixorigin.cn) as alternative, no Docker needed
- **Data dir permission error** → `mkdir -p data/matrixone && chmod 777 data/matrixone`
- **First query slow** → expected with local embedding; model loads into memory on first use (~3-5s). Subsequent queries are fast. Use `--embedding-provider openai` to avoid this.
