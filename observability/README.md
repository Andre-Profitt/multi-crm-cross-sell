# Observability

This directory contains monitoring and observability configurations for the Cross-Sell Intelligence Platform.

## Metrics

The platform exposes Prometheus metrics at `/metrics` endpoint (when enabled).

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by endpoint and status |
| `http_request_duration_seconds` | Histogram | Request latency distribution |
| `model_inference_duration_seconds` | Histogram | ML model inference time |
| `crosssell_recommendations_total` | Counter | Total recommendations generated |
| `crosssell_high_confidence_recommendations` | Gauge | Current high-confidence recommendations |
| `crosssell_model_coverage` | Gauge | Percentage of accounts with recommendations |
| `sync_records_processed_total` | Counter | Records synced from Salesforce by org |
| `celery_task_queue_length` | Gauge | Pending Celery tasks |

## Grafana Dashboard

Import `grafana-dashboard.json` into Grafana to visualize:

1. **API Health**: Success rate, P95 latency, request rate, error rate
2. **ML Pipeline**: Recommendations count, inference latency, model coverage
3. **Data Pipeline**: Sync throughput by org, Celery queue depth
4. **Database**: PostgreSQL connections, database size, Redis status

### Setup

1. Ensure Prometheus is scraping your application:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'crosssell-api'
    static_configs:
      - targets: ['api:8000']
```

2. Add Prometheus as a data source in Grafana

3. Import the dashboard:
   - Go to Dashboards > Import
   - Upload `grafana-dashboard.json` or paste its contents
   - Select your Prometheus data source

## SLOs

Recommended Service Level Objectives:

| SLI | SLO Target | Measurement |
|-----|------------|-------------|
| Availability | 99.9% | `sum(rate(http_requests_total{status!~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |
| Latency P95 | < 500ms | `histogram_quantile(0.95, http_request_duration_seconds_bucket)` |
| Error Rate | < 1% | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |
| Model Inference P95 | < 200ms | `histogram_quantile(0.95, model_inference_duration_seconds_bucket)` |

## Alerting

Example Prometheus alerting rules:

```yaml
groups:
  - name: crosssell
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected

      - alert: SlowInference
        expr: histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le)) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Model inference latency exceeds threshold
```
