FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv
# --system flag installs packages into the system python instead of creating a virtual environment
RUN pipenv install --system --deploy

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
CMD ["python", "scripts/predict.py"]