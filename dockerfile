FROM python:3.9

WORKDIR /app

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Install any additional requirements
RUN pip install uvicorn

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
