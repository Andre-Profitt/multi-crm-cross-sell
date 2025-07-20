# Deployment Guide

This guide describes how to deploy the Multi-CRM Cross-Sell Intelligence Platform in a production environment.

## Using Docker Compose

1. Build the Docker image and start all services:
   ```bash
   docker-compose up --build -d
   ```
2. Run database migrations:
   ```bash
   docker-compose exec app alembic upgrade head
   ```
3. Seed or generate sample data if desired:
   ```bash
   docker-compose exec app python scripts/generate_sample_data.py
   ```
4. Start the API and dashboard containers (already defined in `docker-compose.yml`).
   The API will be available on port `8000` and the dashboard on `8501`.

To stop the stack:
```bash
docker-compose down
```

## Building a Standalone Image

```bash
docker build -t multi-crm-cross-sell:latest .
```
You can push this image to your container registry and run it on any Docker-compatible host.

## Kubernetes (Optional)

For larger deployments you can run the containers on Kubernetes. Build and push the Docker image, then create Kubernetes manifests (Deployments, Services, Secrets, etc.) referencing that image. Apply them with:
```bash
kubectl apply -f k8s/
```
(Replace `k8s/` with the path to your manifests.)

Ensure your environment variables and secret keys are provided via Kubernetes secrets or a vault solution.

## Environment Variables

The application uses the following variables (see `.env.example`):
- `DATABASE_URL`
- `REDIS_URL`
- `SALESFORCE_CLIENT_ID`
- `SALESFORCE_CLIENT_SECRET`
- `JWT_SECRET_KEY`
- `LOG_LEVEL`

Set these values in your deployment environment before starting the containers.
