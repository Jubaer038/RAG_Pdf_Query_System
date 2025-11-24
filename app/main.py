import os
import numpy as np
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gc
import uuid
from pathlib import Path

# Load environment variables from root directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# FastAPI app initialization
app = FastAPI(title="RAG PDF Query API", description="API for querying PDF documents using RAG", version="1.0.0")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to RAG PDF Query API",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/query-pdf/": "POST endpoint to upload PDF and query it"
        },
        "usage": "Visit /docs for interactive API documentation"
    }

# Function to get Hugging Face embeddings
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        raise RuntimeError("No compatible embeddings found. Install `sentence-transformers`.")

# Function to load PDF documents
def load_pdf(pdf_path):
    """Load PDF file and return documents."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

# Function to build the retriever (FAISS vector search)
def build_retriever(documents, embeddings):
    vectorstore = None
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        print(f"FAISS Error: {e}")
        pass

    if vectorstore is not None:
        try:
            if faiss.get_num_gpus() > 0:
                vectorstore.index = faiss.index_cpu_to_all_gpus(vectorstore.index)
            return vectorstore.as_retriever()
        except Exception as e:
            print(f"Error creating retriever: {e}")
            return vectorstore
    return None

# Function to try to get QA chain
def try_get_qa_chain(llm, retriever):
    """Create a QA chain if possible."""
    try:
        # Try multiple import paths to support different langchain versions without static imports
        import importlib
        RetrievalQA = None
        try:
            try:
                mod = importlib.import_module("langchain.chains")
                RetrievalQA = getattr(mod, "RetrievalQA", None)
            except Exception:
                RetrievalQA = None

            if RetrievalQA is None:
                mod = importlib.import_module("langchain.chains.retrieval_qa")
                RetrievalQA = getattr(mod, "RetrievalQA", None)

            if RetrievalQA is None:
                raise ImportError("RetrievalQA could not be imported from langchain")
        except Exception as e:
            raise

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None

# Function to initialize the Hugging Face model (Flan-T5)
def get_huggingface_llm(api_key):
    try:
        from langchain_community.llms import HuggingFaceHub
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            task="text2text-generation",
            model_kwargs={"temperature": 0.1, "max_length": 512},
            huggingfacehub_api_token=api_key
        )
        return llm
    except Exception as e:
        print(f"Error initializing Hugging Face LLM: {e}")
        return None

# Optimized PDF processing with chunking and memory management
def load_and_process_pdf(pdf_path, max_pages=1000):
    """Load and process PDF with memory optimization."""
    documents = load_pdf(pdf_path)
    total_pages = len(documents)
    
    if total_pages > max_pages:
        documents = documents[:max_pages]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=550,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_chunks = text_splitter.split_documents(batch)
        all_chunks.extend(batch_chunks)
        del batch_chunks
        gc.collect()
    
    del documents
    gc.collect()
    
    return all_chunks

# Main query function for running the query
def run_query(query, pdf_path, api_key):
    try:
        documents = load_and_process_pdf(pdf_path)
        embeddings = get_embeddings()
        retriever = build_retriever(documents, embeddings)

        llm = get_huggingface_llm(api_key)
        qa_chain = try_get_qa_chain(llm, retriever)

        if qa_chain:
            answer = qa_chain.run(query)
            return answer
        else:
            if retriever is None:
                raise RuntimeError("No retriever available to fetch documents.")
            # Prefer the standard retriever API if available, otherwise fall back to vectorstore search
            top_docs = None

            # Safely try a typical retriever method
            get_relevant = getattr(retriever, "get_relevant_documents", None)
            if callable(get_relevant):
                try:
                    top_docs = get_relevant(query)
                except TypeError:
                    # Some implementations expect (query, k)
                    try:
                        top_docs = get_relevant(query, k=4)
                    except Exception:
                        top_docs = None

            if top_docs is None:
                # Try to call similarity_search on the retriever itself or its underlying vectorstore/db
                similarity_search_fn = None
                sim = getattr(retriever, "similarity_search", None)
                if callable(sim):
                    similarity_search_fn = sim
                else:
                    vs = getattr(retriever, "vectorstore", None) or getattr(retriever, "db", None)
                    if vs is not None:
                        sim2 = getattr(vs, "similarity_search", None)
                        if callable(sim2):
                            similarity_search_fn = sim2

                if similarity_search_fn is not None:
                    try:
                        top_docs = similarity_search_fn(query, k=4)
                    except TypeError:
                        top_docs = similarity_search_fn(query)
                else:
                    raise RuntimeError("Retriever does not support getting relevant documents.")

            if top_docs and isinstance(top_docs, list):
                return "\n".join([f"--- Passage {i} ---\n{d.page_content[:800]}" for i, d in enumerate(top_docs, 1)])
            else:
                return "No relevant documents found."
    except Exception as e:
        import traceback
        print(f"ERROR in run_query: {e}")
        print(traceback.format_exc())
        return None

# FastAPI Endpoint to handle PDF uploads and queries
@app.post("/query-pdf/")
async def query_pdf(file: UploadFile = File(...), query: str = ""):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required.")
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file temporarily
        file_path = data_dir / f"{uuid.uuid4()}.pdf"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        api_key = os.getenv("HF_API_KEY")
        print(f"DEBUG: API Key loaded: {api_key[:10] if api_key else 'None'}...")
        
        if not api_key: 
            raise HTTPException(status_code=400, detail="API key for Hugging Face is missing. Please set HF_API_KEY in .env file")
        
        # Run the query and process the document
        print(f"DEBUG: Processing query: {query}")
        print(f"DEBUG: PDF file path: {file_path}")
        
        response = run_query(query, str(file_path), api_key)
        
        # Clean up the temporary file
        try:
            file_path.unlink()
        except Exception as cleanup_error:
            print(f"Warning: Could not delete temporary file: {cleanup_error}")
        
        if response:
            return {"answer": response, "query": query}
        else:
            raise HTTPException(status_code=500, detail="Error processing the PDF query. Check server logs for details.")
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Internal server error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
