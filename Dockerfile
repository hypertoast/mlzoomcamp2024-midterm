FROM python:3.10-slim

# Set working directory to scripts directory
WORKDIR /app/scripts

# Copy only the requirements file first
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy files maintaining directory structure
COPY models /app/models
COPY scripts /app/scripts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=9696

# Create non-root user for security
RUN adduser --disabled-password --gecos '' api-user && \
    chown -R api-user:api-user /app
USER api-user

# Expose the port the app runs on
EXPOSE 9696

# Command to run the application
CMD ["python", "predict.py"]