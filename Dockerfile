FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY nlp_api/ ./nlp_api/
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=nlp_api/src/main.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/nlp/health || exit 1

# Create non-root user
RUN useradd --create-home --shell /bin/bash nlp_user
RUN chown -R nlp_user:nlp_user /app
USER nlp_user

# Run the application
CMD ["python", "nlp_api/src/main.py"]

