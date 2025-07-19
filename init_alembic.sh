#!/bin/bash
# Initialize Alembic for database migrations

echo "🔧 Initializing Alembic migrations..."

# Initialize alembic
alembic init migrations

echo "✅ Alembic initialized!"
echo ""
echo "Next steps:"
echo "1. Update alembic.ini with your database URL"
echo "2. Update migrations/env.py to import your models"
echo "3. Create your first migration:"
echo "   alembic revision --autogenerate -m 'Initial schema'"
echo "4. Apply migrations:"
echo "   alembic upgrade head"
