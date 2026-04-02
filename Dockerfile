# Start from Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy your code
COPY hello.py /app/hello.py 

# run your code 
CMD ["python3", "hello.py"] 
