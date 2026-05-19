FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install libraries
RUN pip install -r requirements.txt

# Copy the model files
COPY src/final_pso_cnn_balanced_model.keras /app/
COPY src/pso_scaler.pkl /app/
COPY src/pso_selected_features.pkl /app/ 

#copy the API code
COPY src/api.py /app/ 

#expose port 5000 (tell Docker this app uses port 5000 ) 
EXPOSE 5000

# run your code 
CMD ["python3", "api.py"] 
