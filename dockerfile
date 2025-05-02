# 1. Base Image: Use an official Python slim image
FROM python:3.11-slim

# 2. Environment Variables:
#    - Prevent Python from writing pyc files
#    - Ensure Python output is sent straight to terminal
#    - Set the port Cloud Run expects (default is 8080)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# 3. Working Directory: Set the context for subsequent commands
WORKDIR /app

# 4. Install Dependencies:
#    - Copy only the requirements file first to leverage Docker cache
#    - Install packages specified in requirements.txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code:
#    - Copy the rest of your application code into the container
#    - Use .dockerignore to exclude unnecessary files (like venv, .git)
COPY . .

# 6. Expose Port: Inform Docker that the container listens on this port
EXPOSE 8080

# 7. Run Command: Specify how to start the application
#    - Use Gunicorn to serve the Flask app ('src.app:app' means find 'app' object in src/app.py)
#    - Bind to 0.0.0.0:$PORT to accept connections from outside the container
#    - Set workers/threads as needed (adjust based on Cloud Run instance size)
#    - Set timeout to 0 for potentially long requests (like training, if enabled)
# ... (previous Dockerfile lines remain the same) ...

# 7. Run Command: Use the shell form to allow $PORT substitution
CMD gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 8 --timeout 0 src.app:app