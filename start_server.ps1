# Activate virtual environment and start FastAPI server
& "C:/Users/jubae/RAG SYSTEM/venv/Scripts/Activate.ps1"
Write-Host "Starting FastAPI server..." -ForegroundColor Green
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
