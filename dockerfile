# Use a slim Python base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy only necessary folders
COPY ./app ./app
COPY ./model ./model
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
