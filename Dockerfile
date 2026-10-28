FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Mount a prebuilt index at /app/data (see README), and pass ANTHROPIC_API_KEY at runtime.
EXPOSE 8000
CMD ["biorag", "serve", "--host", "0.0.0.0", "--port", "8000"]
