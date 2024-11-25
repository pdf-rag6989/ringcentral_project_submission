# Use the official slim Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install CA certificates (optional, for TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Disable SSL verification for wandb
ENV WANDB_VERIFY_SSL=false

# Copy all application files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

