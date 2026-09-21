# Using image Python slim version
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements.txt to leverage Docker cache layer
COPY requirements.txt .

# Upgrade pip and install packages without cache
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and necessary files
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI server via uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
