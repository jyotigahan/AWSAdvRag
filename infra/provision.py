"""Provision all AWS resources using boto3. No CLI needed."""
import os
import json
import random
import string
import time
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
BUCKET_NAME = f"rag-documents-{suffix}"
QUEUE_NAME = f"rag-doc-processing-{suffix}"
COLLECTION_NAME = f"rag-search-{suffix}"


def main():
    print("==> Provisioning AWS resources for Advanced RAG...\n")

    # 1. S3 Bucket
    print("==> Creating S3 Bucket...")
    s3 = boto3.client("s3", region_name=REGION)
    create_params = {"Bucket": BUCKET_NAME}
    if REGION != "us-east-1":
        create_params["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
    s3.create_bucket(**create_params)
    # Block public access
    s3.put_public_access_block(
        Bucket=BUCKET_NAME,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"    ✓ {BUCKET_NAME}")

    # 2. SQS Queue
    print("==> Creating SQS Queue...")
    sqs = boto3.client("sqs", region_name=REGION)
    queue_resp = sqs.create_queue(
        QueueName=QUEUE_NAME,
        Attributes={
            "VisibilityTimeout": "900",  # 15 min for document processing
            "MessageRetentionPeriod": "86400",
        },
    )
    queue_url = queue_resp["QueueUrl"]
    print(f"    ✓ {QUEUE_NAME}")

    # 3. OpenSearch Serverless Collection
    print("==> Creating OpenSearch Serverless Collection...")
    aoss = boto3.client("opensearchserverless", region_name=REGION)

    # Create encryption policy
    aoss.create_security_policy(
        name=f"rag-enc-{suffix}",
        type="encryption",
        policy=json.dumps({
            "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]}],
            "AWSOwnedKey": True,
        }),
    )

    # Create network policy (public access for simplicity)
    aoss.create_security_policy(
        name=f"rag-net-{suffix}",
        type="network",
        policy=json.dumps([{
            "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]},
                       {"ResourceType": "dashboard", "Resource": [f"collection/{COLLECTION_NAME}"]}],
            "AllowFromPublic": True,
        }]),
    )

    # Get current caller identity for data access policy
    sts = boto3.client("sts")
    caller = sts.get_caller_identity()
    principal_arn = caller["Arn"]

    # Create data access policy
    aoss.create_access_policy(
        name=f"rag-data-{suffix}",
        type="data",
        policy=json.dumps([{
            "Rules": [
                {"ResourceType": "index", "Resource": [f"index/{COLLECTION_NAME}/*"],
                 "Permission": ["aoss:CreateIndex", "aoss:UpdateIndex", "aoss:DescribeIndex",
                                "aoss:ReadDocument", "aoss:WriteDocument"]},
                {"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"],
                 "Permission": ["aoss:CreateCollectionItems", "aoss:DescribeCollectionItems",
                                "aoss:UpdateCollectionItems"]},
            ],
            "Principal": [principal_arn],
        }]),
    )

    # Create collection
    collection_resp = aoss.create_collection(
        name=COLLECTION_NAME,
        type="VECTORSEARCH",
    )
    collection_id = collection_resp["createCollectionDetail"]["id"]

    # Wait for collection to be active
    print("    Waiting for collection to become active...")
    while True:
        status = aoss.batch_get_collection(ids=[collection_id])
        state = status["collectionDetails"][0]["status"]
        if state == "ACTIVE":
            break
        print(f"    Status: {state}...")
        time.sleep(10)

    endpoint = status["collectionDetails"][0]["collectionEndpoint"]
    print(f"    ✓ {COLLECTION_NAME} ({endpoint})")

    # 4. Verify Bedrock model access
    print("==> Verifying Bedrock model access...")
    bedrock = boto3.client("bedrock", region_name=REGION)
    try:
        models = bedrock.list_foundation_models(byProvider="Anthropic")
        claude_available = any("claude" in m["modelId"] for m in models.get("modelSummaries", []))
        print(f"    ✓ Claude models {'available' if claude_available else 'NOT available — request access in Bedrock console'}")
    except Exception as e:
        print(f"    ⚠ Could not verify: {e}")

    try:
        models = bedrock.list_foundation_models(byProvider="Amazon")
        titan_available = any("titan-embed" in m["modelId"] for m in models.get("modelSummaries", []))
        print(f"    ✓ Titan Embeddings {'available' if titan_available else 'NOT available — request access in Bedrock console'}")
    except Exception as e:
        print(f"    ⚠ Could not verify: {e}")

    # Output
    print("\n=========================================")
    print("  All AWS Resources Provisioned!")
    print("=========================================\n")
    print("Add these to your .env file:\n")
    print(f"AWS_REGION={REGION}")
    print(f"S3_BUCKET_NAME={BUCKET_NAME}")
    print(f"SQS_QUEUE_URL={queue_url}")
    print(f"OPENSEARCH_ENDPOINT={endpoint}")
    print(f"OPENSEARCH_INDEX_NAME=rag-index")
    print(f"BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0")
    print(f"BEDROCK_CHAT_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0")

    # Auto-write .env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(env_path, "w") as f:
        f.write(f"""# Auto-generated by provision.py
AWS_REGION={REGION}

BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_CHAT_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0

OPENSEARCH_ENDPOINT={endpoint}
OPENSEARCH_INDEX_NAME=rag-index

S3_BUCKET_NAME={BUCKET_NAME}
S3_PRESIGNED_EXPIRY=1800

SQS_QUEUE_URL={queue_url}

ACCESS_CONTROL_ENABLED=true
""")
    print(f"\n✓ .env file written to {env_path}")


if __name__ == "__main__":
    main()
