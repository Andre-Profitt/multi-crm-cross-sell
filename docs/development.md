# Development Guide

This document explains how to set up a local development environment for the Multi-CRM Cross-Sell Intelligence Platform.

## Requirements

- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- Docker & Docker Compose (optional but recommended)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Andre-Profitt/multi-crm-cross-sell.git
   cd multi-crm-cross-sell
   ```
2. **Run the setup script** (generates directories and example configs)
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
   Or simply run `make install`.
5. **Copy example configuration**
   ```bash
   cp .env.example .env
   cp config/orgs.example.json config/orgs.json
   ```
   Edit these files with your database credentials and Salesforce settings.

## Running the Application

### Local execution

- Start PostgreSQL and Redis services (or run `make docker-up` to launch them via Docker).
- Run the pipeline once:
  ```bash
  python main.py --run-once
  ```
- Start the API server and dashboard in separate terminals:
  ```bash
  make run-api
  make run-dashboard
  ```

### Using Docker Compose

```bash
# Build images and start services
docker-compose up --build -d

# Run database migrations
docker-compose exec app alembic upgrade head

# Launch the API
docker-compose exec app python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Testing and Linting

- Run all tests:
  ```bash
  make test
  ```
- Lint and format the code:
  ```bash
  make lint
  make format
  ```
