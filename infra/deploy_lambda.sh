#!/bin/bash
set -e

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
FUNCTION_PREFIX="rag-adv"
ROLE_NAME="rag-lambda-role"
S3_BUCKET="${S3_BUCKET_NAME}"
SQS_QUEUE_URL="${SQS_QUEUE_URL}"

echo "==> Creating IAM Role for Lambda..."
TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

ROLE_ARN=$(aws iam create-role \
  --role-name "$ROLE_NAME" \
  --assume-role-policy-document "$TRUST_POLICY" \
  --query Role.Arn --output text 2>/dev/null || \
  aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)

# Attach policies
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

echo "    ✓ $ROLE_NAME"
echo "    Waiting for role propagation..."
sleep 10

echo "==> Packaging Lambda code..."
DEPLOY_DIR=$(mktemp -d)
cp -r ../function_app/* "$DEPLOY_DIR/"
cp -r ../rag/* "$DEPLOY_DIR/"
pip install -r ../requirements.txt -t "$DEPLOY_DIR/" -q
pushd "$DEPLOY_DIR" > /dev/null
zip -r /tmp/rag-lambda.zip . -q
popd > /dev/null
rm -rf "$DEPLOY_DIR"

# Environment variables
ENV_VARS="{\"Variables\":{\"AWS_REGION_NAME\":\"$REGION\",\"S3_BUCKET_NAME\":\"$S3_BUCKET\",\"SQS_QUEUE_URL\":\"$SQS_QUEUE_URL\",\"OPENSEARCH_ENDPOINT\":\"$OPENSEARCH_ENDPOINT\",\"OPENSEARCH_INDEX_NAME\":\"${OPENSEARCH_INDEX_NAME:-rag-index}\",\"BEDROCK_EMBEDDING_MODEL_ID\":\"${BEDROCK_EMBEDDING_MODEL_ID:-amazon.titan-embed-text-v2:0}\",\"BEDROCK_CHAT_MODEL_ID\":\"${BEDROCK_CHAT_MODEL_ID:-anthropic.claude-3-5-sonnet-20241022-v2:0}\"}}"

echo "==> Deploying process-document Lambda..."
aws lambda create-function \
  --function-name "${FUNCTION_PREFIX}-process-doc" \
  --runtime python3.12 \
  --handler lambda_handler.process_document_handler \
  --role "$ROLE_ARN" \
  --zip-file fileb:///tmp/rag-lambda.zip \
  --timeout 900 --memory-size 1024 \
  --environment "$ENV_VARS" \
  --region "$REGION" 2>/dev/null || \
aws lambda update-function-code \
  --function-name "${FUNCTION_PREFIX}-process-doc" \
  --zip-file fileb:///tmp/rag-lambda.zip \
  --region "$REGION"

echo "==> Deploying API Lambda (upload-url, queue, query)..."
aws lambda create-function \
  --function-name "${FUNCTION_PREFIX}-api" \
  --runtime python3.12 \
  --handler api_handler.api_router \
  --role "$ROLE_ARN" \
  --zip-file fileb:///tmp/rag-lambda.zip \
  --timeout 120 --memory-size 512 \
  --environment "$ENV_VARS" \
  --region "$REGION" 2>/dev/null || \
aws lambda update-function-code \
  --function-name "${FUNCTION_PREFIX}-api" \
  --zip-file fileb:///tmp/rag-lambda.zip \
  --region "$REGION"

# Create SQS event source mapping
echo "==> Connecting SQS to process-document Lambda..."
SQS_ARN=$(aws sqs get-queue-attributes --queue-url "$SQS_QUEUE_URL" \
  --attribute-names QueueArn --query "Attributes.QueueArn" --output text)

aws lambda create-event-source-mapping \
  --function-name "${FUNCTION_PREFIX}-process-doc" \
  --event-source-arn "$SQS_ARN" \
  --batch-size 1 2>/dev/null || echo "    (event source mapping already exists)"

echo ""
echo "========================================="
echo "  Lambda Functions Deployed!"
echo "========================================="
echo "  Process: ${FUNCTION_PREFIX}-process-doc"
echo "  API:     ${FUNCTION_PREFIX}-api"
echo ""
echo "Next: Create an API Gateway to expose the API Lambda."
echo ""

rm -f /tmp/rag-lambda.zip
