FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and graphics rendering
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user for Hugging Face Spaces security compliance
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Tell Streamlit to use a directory the 'user' has write access to
ENV STREAMLIT_SERVER_UPLOAD_FILE_DIR=/home/user/tmp

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the temp upload directory and switch ownership
RUN mkdir -p /home/user/tmp && chown -R user:user /home/user/tmp /app
COPY --chown=user . .

# Switch to the non-root user environment
USER user

EXPOSE 8501

# Added CORS and XSRF protection bypass flags for iframe compatibility
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]