FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the correct port
EXPOSE 8080

# Command to run the Gradio app
CMD ["python", "chatbot.py"]