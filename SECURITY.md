# Security Policy

## Data Privacy

- **No real customer data**: This repository contains only synthetic demonstration data
- **Demo dataset**: All sample data is generated using Faker and contains no PII
- **Credentials**: No real Salesforce credentials are included in this repository

## Secrets Management

- Never commit `.env` files (use `.env.example` as template)
- Store Salesforce credentials in environment variables or a secrets manager
- JWT secrets are auto-generated on first run; set `JWT_SECRET_KEY` in production
- API keys should be rotated regularly in production environments

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainer directly with details
3. Allow reasonable time for a fix before disclosure

## Security Best Practices for Deployment

### Authentication
- Change default JWT secret in production
- Use strong passwords for database connections
- Enable rate limiting (configured via `API_RATE_LIMIT`)

### Network
- Run behind a reverse proxy (nginx/traefik) with TLS
- Restrict database access to application containers only
- Use network policies in Kubernetes deployments

### Data
- Encrypt data at rest in production databases
- Use SSL/TLS for all database connections
- Implement audit logging for sensitive operations

## Dependencies

This project uses `detect-secrets` to prevent accidental credential commits. Run pre-commit hooks before pushing:

```bash
pre-commit install
pre-commit run --all-files
```
