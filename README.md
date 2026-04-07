# AWS Advanced RAG Pipeline

## Architecture
```
Upload UI → S3 (presigned URL) → SQS → Lambda (parse/chunk/embed) → OpenSearch Serverless
Query UI → Lambda API → Query Rewrite (Claude) → Hybrid Search (kNN+BM25) → Reranker → Context Manager → Claude → Answer
```

## AWS Services Used
- **Amazon Bedrock** — Claude 3.5 Sonnet (chat) + Titan Embeddings v2 (embeddings)
- **Amazon OpenSearch Serverless** — Vector + text search (hybrid kNN + BM25)
- **Amazon S3** — Document storage with presigned upload URLs
- **Amazon SQS** — Async document processing queue
- **AWS Lambda** — Serverless compute for ingestion + query APIs

## Quick Start

### 1. Provision AWS Resources
```bash
# Option A: Shell script (requires AWS CLI)
chmod +x infra/provision.sh
./infra/provision.sh

# Option B: Python (uses boto3)
python infra/provision.py
```
Copy the output values into `.env` (or it auto-writes).

### 2. Enable Bedrock Model Access
Go to the [Bedrock console](https://console.aws.amazon.com/bedrock/) → Model access → Request access for:
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `amazon.titan-embed-text-v2:0`

### 3. Create OpenSearch Index
```bash
pip install -r requirements.txt
python infra/create_index.py
```

### 4. Deploy Lambda Functions
```bash
chmod +x infra/deploy_lambda.sh
./infra/deploy_lambda.sh
```

### 5. Create API Gateway
Create an HTTP API Gateway in the AWS console pointing to the `rag-adv-api` Lambda, with routes:
- `POST /get-upload-url` → `api_handler.get_upload_url_handler`
- `POST /queue-processing` → `api_handler.queue_processing_handler`
- `POST /query` → `api_handler.query_handler`

Update `API_BASE` in `ui/index.html` and `ui/query.html` with your API Gateway URL.

### 6. Run Query Locally
```bash
cd rag
python -c "
from dotenv import load_dotenv; load_dotenv('../.env')
from chain import rag_query
result = rag_query('What is this document about?')
print(result['answer'])
"
```

### 7. Run Evaluations
```bash
cd evals
python run_evals.py
open report/eval_report.html
```
