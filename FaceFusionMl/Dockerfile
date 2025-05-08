# Use a specific, slim base image. Consider pinning to a digest for reproducibility.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
# Ensure you have a .dockerignore file to exclude unnecessary files/folders
# like .venv, __pycache__, images, test, temp_uploads, etc.
COPY ./app /code/app



# Command to run the application using uvicorn (for production)
# Make sure 'api:app' matches your filename (api.py) and FastAPI app instance name (app)
CMD ["uvicorn", "app.main:app", "--reload","--host", "0.0.0.0", "--port", "8000"]