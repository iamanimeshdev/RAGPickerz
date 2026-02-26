# RAGPickerz

RAGPickerz is a Retrieval-Augmented Generation (RAG) system designed for document analysis and question answering. It was developed for the Bajaj Finserv Hackathon.

## Features

- **Document Loading**: Supports PDF and DOCX file formats
- **Vector Store**: Uses FAISS for efficient document embeddings and similarity search
- **Semantic Chunking**: Splits documents into meaningful chunks using semantic analysis
- **Question Answering**: Processes questions and retrieves relevant answers from documents
- **Batch Processing**: Handles multiple questions concurrently
- **FastAPI Integration**: RESTful API for easy integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/iamanimeshdev/RAGPickerz.git
cd RAGPickerz
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Running the API Server

```bash
# Start the FastAPI server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Making API Requests

#### Endpoint: `/api/v1/hackrx/run`

**Method**: POST
**Content-Type**: multipart/form-data

**Parameters**:
- `questions`: List of questions (JSON array)
- `file`: Document file (PDF or DOCX)

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -F "questions=[\"What is the policy coverage?\", \"What are the exclusions?\"]" \
  -F "file=@document.pdf"
```

**Response**:
```json
{
  "answers": [
    "The policy covers medical expenses up to 5 lakhs.",
    "Pre-existing conditions are excluded from coverage."
  ]
}
```

### Running the Pipeline Directly

```bash
python rag_pipeline/main.py
```

## Configuration

The system uses the following configuration:

- **Vector Database Path**: `.\faiss_index`
- **Chunk Size**: 350 characters
- **Chunk Overlap**: 50 characters
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `gemma-3n-e2b-it`

## File Structure

```
RAGPickerz/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   └── routers/
│       └── hackrx.py        # API router for HackRx endpoints
├── faiss_index/             # Directory for FAISS vector store
├── rag_pipeline/
│   ├── __init__.py
│   ├── config.py           # Configuration and model setup
│   ├── document_loader.py  # Document loading utilities
│   ├── embedder.py         # Vector store building and management
│   ├── query_pipeline.py   # Query processing pipeline
│   ├── retriever.py        # Document retrieval from vector store
│   └── main.py             # Main pipeline execution
├── requirements.txt         # Python dependencies
├── readme.md               # This file
└── .gitignore              # Git ignore rules
```

## Supported Document Formats

- **PDF** (.pdf)
- **Word Documents** (.docx)

## Performance

- **Document Loading**: Optimized for large documents
- **Vector Indexing**: FAISS provides fast similarity search
- **Concurrent Processing**: Handles multiple questions simultaneously
- **Caching**: Vector store is persisted to disk for reuse

## Error Handling

The system includes comprehensive error handling:
- Unsupported file types
- Empty documents
- Rate limiting for API requests
- Server errors with descriptive messages

## Development

### Prerequisites

- Python 3.8+
- pip
- A Google AI Platform account (for LLM access)

### Environment Variables

Create a `.env` file with the following:
```
# Google AI Platform credentials
GOOGLE_API_KEY=your_api_key_here
```

### Testing

```bash
# Run the test suite
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please open an issue on the GitHub repository.