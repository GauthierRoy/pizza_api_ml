# Dockerfile

FROM python:3.12.11-slim

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY ./models/pizza_request_model.joblib ./models/
COPY ./src ./src

# expose the port the app runs on
EXPOSE 8000

# "0.0.0.0" to make it accessible from outside the container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]