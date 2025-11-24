# RAG PDF Query API

A FastAPI-based Retrieval-Augmented Generation (RAG) system for querying PDF documents using AI. This application allows users to upload PDF files and ask questions about their content using natural language processing.

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF files
- **AI-Powered Question Answering**: Ask questions about PDF content using HuggingFace models
- **Vector Search**: Efficient document retrieval using FAISS vector store
- **REST API**: Clean and well-documented API endpoints
- **Interactive Documentation**: Swagger UI and ReDoc for API testing
- **Automatic Reload**: Development server with hot reload support

## 📋 Prerequisites

- Python 3.13 or higher
- Virtual environment (venv)
- HuggingFace API Token

## 🛠️ Installation

### 1. Clone the Repository
```bash
cd "RAG SYSTEM"
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
```

### 3. Activate Virtual Environment
```powershell
& "C:/Users/jubae/RAG SYSTEM/venv/Scripts/Activate.ps1"
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Setup Environment Variables
Create a `.env` file in the root directory:
```env
HF_API_KEY=your_huggingface_api_token_here
```

Get your HuggingFace API token from: https://huggingface.co/settings/tokens

## 📦 Required Packages

The project uses the following main packages:
- `fastapi` - Web framework for building APIs
- `uvicorn` - ASGI server
- `langchain` - LLM framework
- `langchain-community` - Community integrations
- `langchain-text-splitters` - Text splitting utilities
- `sentence-transformers` - Embeddings model
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF processing
- `python-multipart` - File upload support
- `python-dotenv` - Environment variable management

## 🏗️ Project Structure

```
RAG SYSTEM/
├── app/
│   ├── __init__.py
│   ├── main.py              # Main FastAPI application
│   ├── config.py            # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/
│   │           └── __init__.py
│   ├── data/                # Temporary PDF storage
│   │   └── __init__.py
│   ├── model/
│   │   └── __init__.py
│   ├── schemas/
│   │   └── __init__.py
│   ├── services/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── config/
│   └── config.yaml
├── .env                     # Environment variables (create this)
├── requirements.txt         # Python dependencies
├── start_server.ps1        # Server startup script
└── README.md               # This file
```

## 🚀 Running the Application

### Option 1: Using PowerShell Script
```powershell
.\start_server.ps1
```

### Option 2: Manual Start
```powershell
& "C:/Users/jubae/RAG SYSTEM/venv/Scripts/Activate.ps1"
uvicorn app.main:app --reload
```

The server will start at: `http://127.0.0.1:8000`

## 📚 API Endpoints

### Root Endpoint
```
GET /
```
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Welcome to RAG PDF Query API",
  "endpoints": {
    "/docs": "Interactive API documentation",
    "/query-pdf/": "POST endpoint to upload PDF and query it"
  },
  "usage": "Visit /docs for interactive API documentation"
}
```

### Query PDF
```
POST /query-pdf/
```
Upload a PDF file and ask questions about its content.

**Parameters:**
- `file` (form-data): PDF file to upload
- `query` (query parameter): Question to ask about the PDF

**Example using cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/query-pdf/?query=What%20is%20artificial%20intelligence%3F" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@YOUR_FILE.pdf;type=application/pdf"
```

**Response:**
```json
{
  "answer": "Artificial intelligence is...",
  "query": "What is artificial intelligence?"
}
```

## 🌐 Interactive Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 🔧 How It Works

1. **PDF Upload**: User uploads a PDF file through the API
2. **Document Processing**: 
   - PDF is loaded using PyPDFLoader
   - Text is split into chunks using RecursiveCharacterTextSplitter
   - Chunks are processed in batches for memory efficiency
3. **Embedding Generation**: 
   - Text chunks are converted to embeddings using sentence-transformers
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector Storage**: 
   - Embeddings are stored in FAISS vector database
   - Enables fast similarity search
5. **Query Processing**:
   - User query is converted to embeddings
   - Relevant document chunks are retrieved
   - HuggingFace LLM generates answer based on context
6. **Response**: Answer is returned to the user

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'sentence_transformers'**
```powershell
pip install sentence-transformers
```

**2. API Key Error**
- Verify `.env` file exists in the root directory
- Check that `HF_API_KEY` is set correctly
- Get a new token from https://huggingface.co/settings/tokens

**3. Server Won't Start**
- Make sure virtual environment is activated
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

**4. PDF Upload Fails**
- Ensure `data/` directory exists (created automatically)
- Check PDF file is not corrupted
- Verify file size is reasonable

## 📝 Configuration

### Chunk Settings (in main.py)
```python
chunk_size=550        # Characters per chunk
chunk_overlap=200     # Overlap between chunks
batch_size=50         # Chunks processed per batch
max_pages=1000        # Maximum pages to process
```

### Model Configuration
- **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `google/flan-t5-large` (via HuggingFace Hub)
- **Vector Store**: FAISS

## 🔒 Security Notes

- Never commit `.env` file to version control
- Keep your HuggingFace API token secret
- Validate and sanitize file uploads in production
- Implement rate limiting for production use
- Add authentication/authorization as needed

## 🚧 Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT, etc.)
- [ ] Persistent vector store
- [ ] User authentication
- [ ] Chat history
- [ ] Streaming responses
- [ ] Custom model selection
- [ ] Batch processing
- [ ] Docker containerization

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Developed by jubaer

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📞 Support

For issues and questions, please check the terminal logs for detailed error messages.

---

**Last Updated**: November 25, 2025