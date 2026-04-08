"""
Infrastructure as Code — Full AWS Deployment for Banking RAG.

Provisions and deploys everything in one script:
  1. S3 Bucket (documents + static site)
  2. SQS Queue + Dead Letter Queue
  3. OpenSearch Serverless (collection + index)
  4. IAM Roles (Lambda execution)
  5. Lambda Functions (ingestion + API)
  6. API Gateway (HTTP API)
  7. S3 Event → SQS trigger
  8. SQS → Lambda trigger
  9. Static site deployment (UI + Dashboard)
  10. Auto-writes .env

Usage:
    python infra/iac_deploy.py
    python infra/iac_deploy.py --destroy   # tear down everything
"""
import os
import sys
import json
import time
import random
import string
import shutil
import subprocess
import argparse
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
SUFFIX = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
PROJECT = "rag-adv"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Resource names
BUCKET_DOCS = f"{PROJECT}-docs-{SUFFIX}"
BUCKET_SITE = f"{PROJECT}-site-{SUFFIX}"
QUEUE_NAME = f"{PROJECT}-queue-{SUFFIX}"
DLQ_NAME = f"{PROJECT}-dlq-{SUFFIX}"
COLLECTION_NAME = f"{PROJECT}-search-{SUFFIX}"
LAMBDA_INGEST = f"{PROJECT}-ingest-{SUFFIX}"
LAMBDA_API = f"{PROJECT}-api-{SUFFIX}"
ROLE_NAME = f"{PROJECT}-lambda-role-{SUFFIX}"
API_NAME = f"{PROJECT}-gateway-{SUFFIX}"


def get_account_id():
    return boto3.client("sts").get_caller_identity()["Account"]


def get_caller_arn():
    return boto3.client("sts").get_caller_identity()["Arn"]


# ═══════════════════════════════════════════════════════════════
#  STEP 1: S3 BUCKETS
# ═══════════════════════════════════════════════════════════════

def create_s3_buckets():
    s3 = boto3.client("s3", region_name=REGION)

    # Documents bucket
    print(f"  [1/2] Creating documents bucket: {BUCKET_DOCS}")
    params = {"Bucket": BUCKET_DOCS}
    if REGION != "us-east-1":
        params["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
    s3.create_bucket(**params)
    s3.put_public_access_block(Bucket=BUCKET_DOCS, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
    })
    s3.put_bucket_cors(Bucket=BUCKET_DOCS, CORSConfiguration={"CORSRules": [{
        "AllowedOrigins": ["*"], "AllowedMethods": ["GET", "PUT", "POST"],
        "AllowedHeaders": ["*"], "ExposeHeaders": ["ETag"], "MaxAgeSeconds": 3600,
    }]})

    # Static site bucket
    print(f"  [2/2] Creating static site bucket: {BUCKET_SITE}")
    params = {"Bucket": BUCKET_SITE}
    if REGION != "us-east-1":
        params["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
    s3.create_bucket(**params)
    s3.put_bucket_website(Bucket=BUCKET_SITE, WebsiteConfiguration={
        "IndexDocument": {"Suffix": "index.html"},
        "ErrorDocument": {"Key": "index.html"},
    })
    # Allow public read for static site
    s3.delete_public_access_block(Bucket=BUCKET_SITE)
    s3.put_bucket_policy(Bucket=BUCKET_SITE, Policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Sid": "PublicRead", "Effect": "Allow",
                        "Principal": "*", "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{BUCKET_SITE}/*"}],
    }))

    return BUCKET_DOCS, BUCKET_SITE


# ═══════════════════════════════════════════════════════════════
#  STEP 2: SQS QUEUES
# ═══════════════════════════════════════════════════════════════

def create_sqs_queues():
    sqs = boto3.client("sqs", region_name=REGION)

    # Dead letter queue
    print(f"  [1/2] Creating DLQ: {DLQ_NAME}")
    dlq = sqs.create_queue(QueueName=DLQ_NAME, Attributes={"MessageRetentionPeriod": "1209600"})
    dlq_url = dlq["QueueUrl"]
    dlq_arn = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]

    # Main queue with DLQ
    print(f"  [2/2] Creating queue: {QUEUE_NAME}")
    queue = sqs.create_queue(QueueName=QUEUE_NAME, Attributes={
        "VisibilityTimeout": "900",
        "MessageRetentionPeriod": "86400",
        "RedrivePolicy": json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": "3"}),
    })
    queue_url = queue["QueueUrl"]
    queue_arn = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]

    return queue_url, queue_arn, dlq_url


# ═══════════════════════════════════════════════════════════════
#  STEP 3: OPENSEARCH SERVERLESS
# ═══════════════════════════════════════════════════════════════

def create_opensearch():
    aoss = boto3.client("opensearchserverless", region_name=REGION)
    caller_arn = get_caller_arn()

    print(f"  [1/4] Creating encryption policy")
    aoss.create_security_policy(name=f"{PROJECT}-enc-{SUFFIX}", type="encryption",
        policy=json.dumps({"Rules": [{"ResourceType": "collection",
            "Resource": [f"collection/{COLLECTION_NAME}"]}], "AWSOwnedKey": True}))

    print(f"  [2/4] Creating network policy")
    aoss.create_security_policy(name=f"{PROJECT}-net-{SUFFIX}", type="network",
        policy=json.dumps([{"Rules": [
            {"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]},
            {"ResourceType": "dashboard", "Resource": [f"collection/{COLLECTION_NAME}"]}],
            "AllowFromPublic": True}]))

    print(f"  [3/4] Creating data access policy")
    # Include both the caller and the Lambda role
    lambda_role_arn = f"arn:aws:iam::{get_account_id()}:role/{ROLE_NAME}"
    aoss.create_access_policy(name=f"{PROJECT}-data-{SUFFIX}", type="data",
        policy=json.dumps([{"Rules": [
            {"ResourceType": "index", "Resource": [f"index/{COLLECTION_NAME}/*"],
             "Permission": ["aoss:CreateIndex", "aoss:UpdateIndex", "aoss:DescribeIndex",
                            "aoss:ReadDocument", "aoss:WriteDocument"]},
            {"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"],
             "Permission": ["aoss:CreateCollectionItems", "aoss:DescribeCollectionItems",
                            "aoss:UpdateCollectionItems"]}],
            "Principal": [caller_arn, lambda_role_arn]}]))

    print(f"  [4/4] Creating collection: {COLLECTION_NAME}")
    resp = aoss.create_collection(name=COLLECTION_NAME, type="VECTORSEARCH")
    collection_id = resp["createCollectionDetail"]["id"]

    print(f"        Waiting for collection to become active...")
    while True:
        status = aoss.batch_get_collection(ids=[collection_id])
        state = status["collectionDetails"][0]["status"]
        if state == "ACTIVE":
            break
        print(f"        Status: {state}...")
        time.sleep(10)

    endpoint = status["collectionDetails"][0]["collectionEndpoint"]
    print(f"        ✓ {COLLECTION_NAME} ({endpoint})")
    return endpoint, collection_id


# ═══════════════════════════════════════════════════════════════
#  STEP 4: CREATE OPENSEARCH INDEX
# ═══════════════════════════════════════════════════════════════

def create_search_index(endpoint):
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    print(f"  Creating rag-index on {endpoint}")
    creds = boto3.Session().get_credentials()
    auth = AWS4Auth(creds.access_key, creds.secret_key, REGION, "aoss", session_token=creds.token)
    host = endpoint.replace("https://", "")
    client = OpenSearch(hosts=[{"host": host, "port": 443}], http_auth=auth,
                        use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection)

    index_body = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
        "mappings": {"properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "content_vector": {"type": "knn_vector", "dimension": 1024,
                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "nmslib",
                           "parameters": {"ef_construction": 512, "m": 16}}},
            "source": {"type": "keyword"}, "department": {"type": "keyword"},
            "access_level": {"type": "keyword"}, "author": {"type": "keyword"},
            "page": {"type": "integer"}, "chunk_index": {"type": "integer"},
            "total_chunks": {"type": "integer"}, "date_uploaded": {"type": "date"},
        }},
    }

    if client.indices.exists(index="rag-index"):
        client.indices.delete(index="rag-index")
        time.sleep(2)
    client.indices.create(index="rag-index", body=index_body)
    print(f"        ✓ rag-index created")


# ═══════════════════════════════════════════════════════════════
#  STEP 5: IAM ROLE FOR LAMBDA
# ═══════════════════════════════════════════════════════════════

def create_lambda_role():
    iam = boto3.client("iam")
    account_id = get_account_id()

    print(f"  [1/2] Creating role: {ROLE_NAME}")
    role = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole"}],
    }))
    role_arn = role["Role"]["Arn"]

    print(f"  [2/2] Attaching policies")
    # Inline policy with all needed permissions
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="rag-lambda-permissions",
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "*"},
            {"Effect": "Allow", "Action": "s3:*", "Resource": [f"arn:aws:s3:::{BUCKET_DOCS}", f"arn:aws:s3:::{BUCKET_DOCS}/*"]},
            {"Effect": "Allow", "Action": "sqs:*", "Resource": f"arn:aws:sqs:{REGION}:{account_id}:{QUEUE_NAME}"},
            {"Effect": "Allow", "Action": ["bedrock:*", "bedrock-runtime:*"], "Resource": "*"},
            {"Effect": "Allow", "Action": "aoss:*", "Resource": "*"},
            {"Effect": "Allow", "Action": ["aws-marketplace:ViewSubscriptions", "aws-marketplace:Subscribe"], "Resource": "*"},
        ]}))

    print(f"        Waiting 10s for IAM propagation...")
    time.sleep(10)
    return role_arn


# ═══════════════════════════════════════════════════════════════
#  STEP 6: PACKAGE & DEPLOY LAMBDA FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def package_lambda():
    print(f"  Packaging Lambda code...")
    deploy_dir = os.path.join(BASE_DIR, "infra", "_lambda_pkg")
    zip_path = os.path.join(BASE_DIR, "infra", "lambda_package.zip")

    # Clean
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)

    # Copy code — flatten all modules into root
    for folder in ["function_app", "rag"]:
        src = os.path.join(BASE_DIR, folder)
        if os.path.exists(src):
            for f in os.listdir(src):
                src_file = os.path.join(src, f)
                if os.path.isfile(src_file) and f.endswith(".py"):
                    shutil.copy2(src_file, os.path.join(deploy_dir, f))

    # Install ONLY the deps Lambda needs (no sentence-transformers, no torch)
    # Use --platform to get Linux binaries for Lambda
    lambda_deps = [
        "opensearch-py>=2.4.0,<3.0.0",
        "requests-aws4auth",
        "tiktoken",
        "pypdf",
        "python-docx",
        "lxml",
        "requests",
        "urllib3",
        "certifi",
        "python-dateutil",
        "six",
        "regex",
        "typing_extensions",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + lambda_deps +
                    ["-t", deploy_dir, "-q",
                     "--platform", "manylinux2014_x86_64",
                     "--implementation", "cp",
                     "--python-version", "3.12",
                     "--only-binary=:all:",
                     "--upgrade"], check=False)
    # Install pure-python packages that don't have binary wheels
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "opensearch-py>=2.4.0,<3.0.0", "requests-aws4auth", "pypdf", "python-docx",
                    "-t", deploy_dir, "-q", "--upgrade", "--no-deps"], check=False)

    # Zip
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", deploy_dir)

    # Clean up
    shutil.rmtree(deploy_dir)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"        ✓ Package: {size_mb:.1f} MB")

    # If > 50MB, upload to S3 and return S3 key
    if size_mb > 49:
        s3 = boto3.client("s3", region_name=REGION)
        s3_key = "lambda/lambda_package.zip"
        print(f"        Package > 50MB, uploading to S3...")
        s3.upload_file(zip_path, BUCKET_DOCS, s3_key)
        os.remove(zip_path)
        print(f"        ✓ Uploaded to s3://{BUCKET_DOCS}/{s3_key}")
        return {"S3Bucket": BUCKET_DOCS, "S3Key": s3_key}

    return zip_path


def deploy_lambdas(role_arn, code_location, queue_url, opensearch_endpoint):
    lam = boto3.client("lambda", region_name=REGION)

    env_vars = {
        "AWS_REGION_NAME": REGION,
        "S3_BUCKET_NAME": BUCKET_DOCS,
        "SQS_QUEUE_URL": queue_url,
        "OPENSEARCH_ENDPOINT": opensearch_endpoint,
        "OPENSEARCH_INDEX_NAME": "rag-index",
        "BEDROCK_EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
        "BEDROCK_CHAT_MODEL_ID": "us.amazon.nova-micro-v1:0",
    }

    # Determine code source
    if isinstance(code_location, dict):
        code = code_location  # S3 reference
    else:
        with open(code_location, "rb") as f:
            code = {"ZipFile": f.read()}

    functions = [
        (LAMBDA_INGEST, "lambda_handler.process_document_handler", 900, 1024),
        (LAMBDA_API, "api_handler.query_handler", 120, 512),
        (f"{LAMBDA_API}-upload", "api_handler.get_upload_url_handler", 30, 256),
        (f"{LAMBDA_API}-queue", "api_handler.queue_processing_handler", 30, 256),
    ]

    for i, (name, handler, timeout, memory) in enumerate(functions):
        print(f"  [{i+1}/{len(functions)}] Deploying: {name}")
        lam.create_function(
            FunctionName=name, Runtime="python3.12", Handler=handler,
            Role=role_arn, Code=code, Timeout=timeout, MemorySize=memory,
            Environment={"Variables": env_vars},
        )

    return LAMBDA_INGEST, LAMBDA_API


# ═══════════════════════════════════════════════════════════════
#  STEP 7: SQS → LAMBDA TRIGGER
# ═══════════════════════════════════════════════════════════════

def create_sqs_trigger(queue_arn):
    lam = boto3.client("lambda", region_name=REGION)
    print(f"  Connecting SQS → {LAMBDA_INGEST}")
    lam.create_event_source_mapping(
        FunctionName=LAMBDA_INGEST,
        EventSourceArn=queue_arn,
        BatchSize=1,
        Enabled=True,
    )


# ═══════════════════════════════════════════════════════════════
#  STEP 8: S3 EVENT → SQS (auto-ingest on upload)
# ═══════════════════════════════════════════════════════════════

def create_s3_notification(queue_arn, queue_url):
    s3 = boto3.client("s3", region_name=REGION)
    sqs = boto3.client("sqs", region_name=REGION)
    account_id = get_account_id()

    # Allow S3 to send messages to SQS
    sqs.set_queue_attributes(QueueUrl=queue_url, Attributes={
        "Policy": json.dumps({
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Principal": {"Service": "s3.amazonaws.com"},
                           "Action": "sqs:SendMessage", "Resource": queue_arn,
                           "Condition": {"ArnLike": {"aws:SourceArn": f"arn:aws:s3:::{BUCKET_DOCS}"}}}],
        })
    })

    # S3 event notification
    s3.put_bucket_notification_configuration(Bucket=BUCKET_DOCS, NotificationConfiguration={
        "QueueConfigurations": [{
            "QueueArn": queue_arn,
            "Events": ["s3:ObjectCreated:*"],
        }]
    })
    print(f"  ✓ S3 upload → SQS → Lambda auto-ingest enabled")


# ═══════════════════════════════════════════════════════════════
#  STEP 9: API GATEWAY
# ═══════════════════════════════════════════════════════════════

def create_api_gateway():
    apigw = boto3.client("apigatewayv2", region_name=REGION)
    lam = boto3.client("lambda", region_name=REGION)
    account_id = get_account_id()

    print(f"  [1/4] Creating HTTP API: {API_NAME}")
    api = apigw.create_api(
        Name=API_NAME, ProtocolType="HTTP",
        CorsConfiguration={
            "AllowOrigins": ["*"], "AllowMethods": ["GET", "POST", "OPTIONS"],
            "AllowHeaders": ["Content-Type"],
        },
    )
    api_id = api["ApiId"]

    # Create integrations for each Lambda
    routes = {
        "POST /query": LAMBDA_API,
        "POST /get-upload-url": f"{LAMBDA_API}-upload",
        "POST /queue-processing": f"{LAMBDA_API}-queue",
    }

    for i, (route_key, func_name) in enumerate(routes.items()):
        print(f"  [{i+2}/4] Route: {route_key} → {func_name}")
        func_arn = lam.get_function(FunctionName=func_name)["Configuration"]["FunctionArn"]

        integration = apigw.create_integration(
            ApiId=api_id, IntegrationType="AWS_PROXY",
            IntegrationUri=func_arn, PayloadFormatVersion="2.0",
        )

        apigw.create_route(
            ApiId=api_id, RouteKey=route_key,
            Target=f"integrations/{integration['IntegrationId']}",
        )

        # Grant API Gateway permission to invoke Lambda
        try:
            lam.add_permission(
                FunctionName=func_name, StatementId=f"apigw-{route_key.replace(' ', '-').replace('/', '-')}",
                Action="lambda:InvokeFunction", Principal="apigateway.amazonaws.com",
                SourceArn=f"arn:aws:execute-api:{REGION}:{account_id}:{api_id}/*",
            )
        except lam.exceptions.ResourceConflictException:
            pass

    # Deploy
    apigw.create_stage(ApiId=api_id, StageName="prod", AutoDeploy=True)
    api_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/prod"
    print(f"        ✓ API URL: {api_url}")
    return api_id, api_url


# ═══════════════════════════════════════════════════════════════
#  STEP 10: DEPLOY STATIC SITE (UI + Dashboard)
# ═══════════════════════════════════════════════════════════════

def deploy_static_site(api_url):
    s3 = boto3.client("s3", region_name=REGION)
    site_url = f"http://{BUCKET_SITE}.s3-website-{REGION}.amazonaws.com"

    print(f"  Deploying UI to {BUCKET_SITE}")

    # Upload UI files with API_BASE replaced
    ui_dir = os.path.join(BASE_DIR, "ui")
    for filename in ["index.html", "query.html", "styles.css"]:
        filepath = os.path.join(ui_dir, filename)
        with open(filepath, "r") as f:
            content = f.read()

        # Replace API_BASE for production
        content = content.replace(
            'window.location.origin + "/api"',
            f'"{api_url}"'
        )

        content_type = "text/html" if filename.endswith(".html") else "text/css"
        s3.put_object(Bucket=BUCKET_SITE, Key=filename, Body=content.encode(),
                      ContentType=content_type)

    # Upload dashboard
    dash_dir = os.path.join(BASE_DIR, "dashboard")
    for filename in ["dashboard.html"]:
        filepath = os.path.join(dash_dir, filename)
        with open(filepath, "r") as f:
            content = f.read()
        s3.put_object(Bucket=BUCKET_SITE, Key=filename, Body=content.encode(),
                      ContentType="text/html")

    print(f"        ✓ Site URL: {site_url}")
    return site_url


# ═══════════════════════════════════════════════════════════════
#  STEP 11: WRITE .ENV + DEPLOYMENT MANIFEST
# ═══════════════════════════════════════════════════════════════

def write_env(queue_url, opensearch_endpoint, api_url, site_url):
    env_path = os.path.join(BASE_DIR, ".env")
    with open(env_path, "w") as f:
        f.write(f"""# Auto-generated by iac_deploy.py — {time.strftime('%Y-%m-%d %H:%M:%S')}
AWS_REGION={REGION}

BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_CHAT_MODEL_ID=us.amazon.nova-micro-v1:0

OPENSEARCH_ENDPOINT={opensearch_endpoint}
OPENSEARCH_INDEX_NAME=rag-index

S3_BUCKET_NAME={BUCKET_DOCS}
S3_PRESIGNED_EXPIRY=1800

SQS_QUEUE_URL={queue_url}

ACCESS_CONTROL_ENABLED=true
""")

    # Deployment manifest
    manifest = {
        "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "region": REGION,
        "suffix": SUFFIX,
        "resources": {
            "s3_docs": BUCKET_DOCS,
            "s3_site": BUCKET_SITE,
            "sqs_queue": queue_url,
            "sqs_dlq": DLQ_NAME,
            "opensearch_collection": COLLECTION_NAME,
            "opensearch_endpoint": opensearch_endpoint,
            "lambda_ingest": LAMBDA_INGEST,
            "lambda_api": LAMBDA_API,
            "lambda_api_upload": f"{LAMBDA_API}-upload",
            "lambda_api_queue": f"{LAMBDA_API}-queue",
            "api_gateway": API_NAME,
            "api_url": api_url,
            "site_url": site_url,
            "iam_role": ROLE_NAME,
        },
        "urls": {
            "upload_ui": f"{site_url}/index.html",
            "query_ui": f"{site_url}/query.html",
            "dashboard": f"{site_url}/dashboard.html",
            "api_query": f"{api_url}/query",
            "api_upload": f"{api_url}/get-upload-url",
        },
    }
    manifest_path = os.path.join(BASE_DIR, "infra", "deployment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  ✓ .env written")
    print(f"  ✓ Manifest: infra/deployment_manifest.json")
    return manifest


# ═══════════════════════════════════════════════════════════════
#  DESTROY — Tear down everything
# ═══════════════════════════════════════════════════════════════

def destroy():
    manifest_path = os.path.join(BASE_DIR, "infra", "deployment_manifest.json")
    if not os.path.exists(manifest_path):
        print("No deployment manifest found. Nothing to destroy.")
        return

    with open(manifest_path) as f:
        m = json.load(f)
    res = m["resources"]

    print("\n==> DESTROYING all AWS resources...\n")

    # Lambda event source mappings
    lam = boto3.client("lambda", region_name=REGION)
    try:
        mappings = lam.list_event_source_mappings(FunctionName=res["lambda_ingest"])
        for mapping in mappings.get("EventSourceMappings", []):
            lam.delete_event_source_mapping(UUID=mapping["UUID"])
            print(f"  Deleted event source mapping: {mapping['UUID']}")
    except Exception:
        pass

    # API Gateway
    try:
        apigw = boto3.client("apigatewayv2", region_name=REGION)
        apis = apigw.get_apis()["Items"]
        for api in apis:
            if api["Name"] == res.get("api_gateway", API_NAME):
                apigw.delete_api(ApiId=api["ApiId"])
                print(f"  Deleted API Gateway: {api['Name']}")
    except Exception as e:
        print(f"  API Gateway cleanup: {e}")

    # Lambdas
    for func in [res.get("lambda_ingest"), res.get("lambda_api"),
                 res.get("lambda_api_upload"), res.get("lambda_api_queue")]:
        if func:
            try:
                lam.delete_function(FunctionName=func)
                print(f"  Deleted Lambda: {func}")
            except Exception:
                pass

    # IAM Role
    iam = boto3.client("iam")
    try:
        iam.delete_role_policy(RoleName=res["iam_role"], PolicyName="rag-lambda-permissions")
        iam.delete_role(RoleName=res["iam_role"])
        print(f"  Deleted IAM role: {res['iam_role']}")
    except Exception as e:
        print(f"  IAM cleanup: {e}")

    # S3 buckets
    s3 = boto3.resource("s3", region_name=REGION)
    for bucket_name in [res.get("s3_docs"), res.get("s3_site")]:
        if bucket_name:
            try:
                bucket = s3.Bucket(bucket_name)
                bucket.objects.all().delete()
                bucket.delete()
                print(f"  Deleted S3: {bucket_name}")
            except Exception as e:
                print(f"  S3 cleanup {bucket_name}: {e}")

    # SQS
    sqs = boto3.client("sqs", region_name=REGION)
    for q_name in [QUEUE_NAME, DLQ_NAME]:
        try:
            url = sqs.get_queue_url(QueueName=q_name)["QueueUrl"]
            sqs.delete_queue(QueueUrl=url)
            print(f"  Deleted SQS: {q_name}")
        except Exception:
            pass
    # Try from manifest
    try:
        sqs.delete_queue(QueueUrl=res["sqs_queue"])
    except Exception:
        pass

    # OpenSearch
    aoss = boto3.client("opensearchserverless", region_name=REGION)
    sfx = m.get("suffix", "")
    try:
        colls = aoss.list_collections()["collectionSummaries"]
        for c in colls:
            if sfx in c["name"]:
                aoss.delete_collection(id=c["id"])
                print(f"  Deleted OpenSearch collection: {c['name']}")
    except Exception as e:
        print(f"  OpenSearch cleanup: {e}")

    # OpenSearch policies
    for ptype in ["encryption", "network"]:
        try:
            policies = aoss.list_security_policies(type=ptype)["securityPolicySummaries"]
            for p in policies:
                if sfx in p["name"]:
                    aoss.delete_security_policy(name=p["name"], type=ptype)
                    print(f"  Deleted {ptype} policy: {p['name']}")
        except Exception:
            pass
    try:
        policies = aoss.list_access_policies(type="data")["accessPolicySummaries"]
        for p in policies:
            if sfx in p["name"]:
                aoss.delete_access_policy(name=p["name"], type="data")
                print(f"  Deleted data policy: {p['name']}")
    except Exception:
        pass

    os.remove(manifest_path)
    print(f"\n✓ All resources destroyed. Manifest removed.")


# ═══════════════════════════════════════════════════════════════
#  MAIN — Orchestrate everything
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IaC Deploy for Banking RAG")
    parser.add_argument("--destroy", action="store_true", help="Tear down all resources")
    args = parser.parse_args()

    if args.destroy:
        destroy()
        return

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          AWS BANKING RAG — FULL DEPLOYMENT                  ║
║          Region: {REGION:<20s} Suffix: {SUFFIX:<10s}       ║
╚══════════════════════════════════════════════════════════════╝
""")

    print("STEP 1: S3 Buckets")
    bucket_docs, bucket_site = create_s3_buckets()

    print("\nSTEP 2: SQS Queues")
    queue_url, queue_arn, dlq_url = create_sqs_queues()

    print("\nSTEP 3: IAM Role")
    role_arn = create_lambda_role()

    print("\nSTEP 4: OpenSearch Serverless")
    opensearch_endpoint, collection_id = create_opensearch()

    print("\nSTEP 5: OpenSearch Index")
    create_search_index(opensearch_endpoint)

    print("\nSTEP 6: Package Lambda")
    code_location = package_lambda()

    print("\nSTEP 7: Deploy Lambda Functions")
    deploy_lambdas(role_arn, code_location, queue_url, opensearch_endpoint)

    print("\nSTEP 8: SQS → Lambda Trigger")
    create_sqs_trigger(queue_arn)

    print("\nSTEP 9: S3 Upload → SQS Notification")
    create_s3_notification(queue_arn, queue_url)

    print("\nSTEP 10: API Gateway")
    api_id, api_url = create_api_gateway()

    print("\nSTEP 11: Deploy Static Site")
    site_url = deploy_static_site(api_url)

    print("\nSTEP 12: Write Config")
    manifest = write_env(queue_url, opensearch_endpoint, api_url, site_url)

    # Clean up zip if local
    zip_path = os.path.join(BASE_DIR, "infra", "lambda_package.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    DEPLOYMENT COMPLETE                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Upload UI:   {manifest['urls']['upload_ui']:<44s}║
║  Query UI:    {manifest['urls']['query_ui']:<44s}║
║  Dashboard:   {manifest['urls']['dashboard']:<44s}║
║                                                              ║
║  API:         {api_url:<44s}║
║                                                              ║
║  To destroy:  python infra/iac_deploy.py --destroy           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
