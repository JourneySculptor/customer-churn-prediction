FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the essential files first to leverage Docker caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the model/scaler files
COPY . /app

# Expose the API port
EXPOSE 8080

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
