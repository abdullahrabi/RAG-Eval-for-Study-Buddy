FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
# Run the Python script directly, not with streamlit run
CMD ["python", "entrypoint.py"]