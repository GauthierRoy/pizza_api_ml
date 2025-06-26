# Dockerfile

# 1. Use an official, lightweight Python base image
FROM python:3.12.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file FIRST to leverage Docker's layer caching
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Expose the port the app runs on (informational)
EXPOSE 8000

# 7. The command to run your application when the container starts
# Use "0.0.0.0" to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]