# Deployment Guide

This project can be deployed to a Kubernetes cluster. Example manifests live in the [`k8s/`](../k8s/) directory.

## Build and Push the Image

```bash
docker build -t your-registry/multi-crm-cross-sell:latest .
docker push your-registry/multi-crm-cross-sell:latest
```

## Apply Manifests

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

Ensure an ingress controller is installed to expose the service.

## Monitoring with Helm

Helm value files are provided under `k8s/monitoring/` for Prometheus and Grafana.
Install them with:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack -f k8s/monitoring/prometheus-values.yaml
helm install grafana grafana/grafana -f k8s/monitoring/grafana-values.yaml
```
