# Start from Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install libraries
RUN pip install -r requirements.txt

# Copy your code
COPY src/ /app/ 

# run your code 
CMD ["python3", "main.py"] 
