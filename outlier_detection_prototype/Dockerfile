FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for hdbscan and other packages that might need compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the data directory exists
RUN mkdir -p data

# Expose ports for FastAPI (8000) and Streamlit (8001)
EXPOSE 8000
EXPOSE 8001

# By default, run the pipeline to ensure data is available,
# but this will likely be overridden in docker-compose.yml
CMD ["python", "src/pipeline.py"]
