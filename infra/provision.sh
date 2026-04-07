#!/bin/bash
set -e

REGION="${AWS_REGION:-us-east-1}"
SUFFIX=$(openssl rand -hex 4)
BUCKET_NAME="rag-documents-${SUFFIX}"
QUEUE_NAME="rag-doc-processing-${SUFFIX}"
COLLECTION_NAME="rag-search-${SUFFIX}"

echo "==> Creating S3 Bucket..."
if [ "$REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
else
  aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"
fi
aws s3api put-public-access-block --bucket "$BUCKET_NAME" \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
echo "    ✓ $BUCKET_NAME"

echo "==> Creating SQS Queue..."
QUEUE_URL=$(aws sqs create-queue --queue-name "$QUEUE_NAME" \
  --attributes VisibilityTimeout=900,MessageRetentionPeriod=86400 \
  --query QueueUrl --output text)
echo "    ✓ $QUEUE_NAME"

echo "==> Creating OpenSearch Serverless Collection..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text)

# Encryption policy
aws opensearchserverless create-security-policy \
  --name "rag-enc-${SUFFIX}" --type encryption \
  --policy "{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${COLLECTION_NAME}\"]}],\"AWSOwnedKey\":true}"

# Network policy
aws opensearchserverless create-security-policy \
  --name "rag-net-${SUFFIX}" --type network \
  --policy "[{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${COLLECTION_NAME}\"]},{\"ResourceType\":\"dashboard\",\"Resource\":[\"collection/${COLLECTION_NAME}\"]}],\"AllowFromPublic\":true}]"

# Data access policy
aws opensearchserverless create-access-policy \
  --name "rag-data-${SUFFIX}" --type data \
  --policy "[{\"Rules\":[{\"ResourceType\":\"index\",\"Resource\":[\"index/${COLLECTION_NAME}/*\"],\"Permission\":[\"aoss:CreateIndex\",\"aoss:UpdateIndex\",\"aoss:DescribeIndex\",\"aoss:ReadDocument\",\"aoss:WriteDocument\"]},{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${COLLECTION_NAME}\"],\"Permission\":[\"aoss:CreateCollectionItems\",\"aoss:DescribeCollectionItems\",\"aoss:UpdateCollectionItems\"]}],\"Principal\":[\"${CALLER_ARN}\"]}]"

# Create collection
COLLECTION_ID=$(aws opensearchserverless create-collection \
  --name "$COLLECTION_NAME" --type VECTORSEARCH \
  --query "createCollectionDetail.id" --output text)

echo "    Waiting for collection to become active..."
while true; do
  STATUS=$(aws opensearchserverless batch-get-collection \
    --ids "$COLLECTION_ID" --query "collectionDetails[0].status" --output text)
  if [ "$STATUS" = "ACTIVE" ]; then break; fi
  echo "    Status: $STATUS..."
  sleep 10
done

ENDPOINT=$(aws opensearchserverless batch-get-collection \
  --ids "$COLLECTION_ID" --query "collectionDetails[0].collectionEndpoint" --output text)
echo "    ✓ $COLLECTION_NAME ($ENDPOINT)"

echo ""
echo "========================================="
echo "  AWS Resources Provisioned!"
echo "========================================="
echo ""
echo "Add these to your .env file:"
echo ""
echo "AWS_REGION=$REGION"
echo "S3_BUCKET_NAME=$BUCKET_NAME"
echo "SQS_QUEUE_URL=$QUEUE_URL"
echo "OPENSEARCH_ENDPOINT=$ENDPOINT"
echo "OPENSEARCH_INDEX_NAME=rag-index"
echo "BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0"
echo "BEDROCK_CHAT_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0"
echo ""
