
# PromiseTrack AI

PromiseTrack AI is an advanced artificial intelligence system that tracks forward-looking management commitments from corporate transcripts and measures whether they materialize in real financial outcomes. 

By utilizing Natural Language Processing (NLP), Machine Learning, and Retrieval-Augmented Generation (RAG), PromiseTrack AI automatically extracts claims made by company executives, parses financial data to verify these claims, and provides comprehensive analysis including risk assessments and consistency scores.

## Key Features

- **Text Extraction Pipeline:** Parses corporate transcripts (PDFs) and splits them into processable sentences.
- **Claim Extraction:** Uses DistilBERT and SpaCy pipelines to identify and extract forward-looking management claims and promises.
- **Financial Data Parsing:** Extracts quarterly financial data from local XBRL/XML files.
- **Verification Engine:** Cross-references extracted NLP claims against actual financial timeseries data to verify accuracy and compute consistency scores.
- **Risk Assessment:** Evaluates the consistency of management's statements and flags potential high-risk corporate communications.
- **RAG Explainer (ChromaDB + Groq):** Provides detailed analysis, positive/negative signals, and overall verdicts on a company's financial performance by utilizing vector embeddings.
- **Flask Backend:** A web backend to serve the application and process pipelines.

## Project Structure

- `app/`: Flask web backend providing routes and cache services.
- `pipelines/`: Contains the core extraction and processing pipelines.
  - `text/`: Text extraction, sentence splitting, and claim extraction.
  - `ml/`: Machine learning models for attribute extraction.
  - `finance/`: XBRL data extraction and timeseries preparation.
  - `risk/`: Aggregates risk metrics based on verified claims.
  - `rag/`: Vector DB builder (ChromaDB) and explanations via LLM.
- `data/`: Local storage for raw transcripts and financial data.
- `process_company.py`: Single-entry CLI script to process companies locally.
- `run.py`: Entry point for the Flask application.

## Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Configure your environment variables in a `.env` file or update `config.py`. Ensure you have your keys set up for services like Groq.

3. **Running the Pipeline:**
   To process a specific company's transcripts and XBRL data:
   ```bash
   python process_company.py "Company Name"
   ```
   To process all available companies:
   ```bash
   python process_company.py --all
   ```

4. **Running the Web App:**
   Start the Flask application:
   ```bash
   python run.py
   ```

## Processing Workflow

1. Transcript text is extracted and chunked.
2. The claim extraction model identifies commitments.
3. Financial pipeline pulls XBRL data and generates a timeseries.
4. Claims are verified against the timeseries data.
5. All text and results are loaded into ChromaDB.
6. The RAG pipeline generates detailed positive/negative signals and explanations.
