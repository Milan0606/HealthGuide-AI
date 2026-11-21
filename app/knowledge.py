import json
import time
from functools import lru_cache

import boto3
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

EMBEDDING_DIMENSION = settings.embed_dim


@lru_cache()
def get_bedrock_client():
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        raise ValueError("AWS credentials must be set.")
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


@lru_cache()
def get_pinecone_client():
    if not settings.pinecone_api_key:
        raise ValueError("Pinecone API key must be set.")
    return Pinecone(api_key=settings.pinecone_api_key)


def clients():
    return get_bedrock_client(), get_pinecone_client()


def ensure_index():
    pc = get_pinecone_client()

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(settings.pinecone_index)


def _embed_single(client, text: str, max_retries: int = 5):
    body = json.dumps({
        "inputText": text,
        "dimensions": settings.embed_dim,
        "normalize": True
    })

    for attempt in range(1, max_retries + 1):
        try:
            response = client.invoke_model(
                modelId=settings.bedrock_embed_model,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]
        except Exception as e:
            if attempt == max_retries:
                raise
            wait_time = 2 ** attempt
            print(f"[knowledge] Bedrock embedding request failed: {e}. Retrying in {wait_time}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait_time)


def embed(texts, input_type="passage"):
    client = get_bedrock_client()
    return [_embed_single(client, text) for text in texts]


def search_kb(query, top_k=5):
    index = ensure_index()

    # User query embedding
    vector = embed([query], input_type="query")[0]

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "page": match.metadata.get("page"),
            "text": match.metadata.get("text", ""),
            "score": round(match.score, 3),
            "source": "Gale Encyclopedia (2002)"
        }
        for match in result.matches
        if match.score >= 0.30
    ]
