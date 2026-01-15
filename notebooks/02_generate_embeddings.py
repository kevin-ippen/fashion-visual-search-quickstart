# Databricks notebook source
# MAGIC %md
# MAGIC # 02: Generate FashionCLIP Embeddings
# MAGIC
# MAGIC This notebook generates 512-dimensional visual embeddings for each product using FashionCLIP.
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Loads product images from Unity Catalog Volume
# MAGIC 2. Converts images to base64 format
# MAGIC 3. Calls the FashionCLIP model serving endpoint in batches
# MAGIC 4. Saves embeddings to a Delta table
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Completed notebook 01 (products table exists)
# MAGIC - FashionCLIP endpoint deployed and running
# MAGIC - Product images uploaded to UC Volume
# MAGIC
# MAGIC **Runtime:** CPU cluster (GPU not required - endpoint handles inference)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration - UPDATE THESE
CATALOG = "main"
SCHEMA = "fashion_quickstart"
EMBEDDING_ENDPOINT = "fashionclip-endpoint"

# Processing settings
BATCH_SIZE = 32
REQUEST_TIMEOUT = 120  # seconds

# Tables
PRODUCTS_TABLE = f"{CATALOG}.{SCHEMA}.products"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.product_embeddings"

print(f"Products table: {PRODUCTS_TABLE}")
print(f"Embeddings table: {EMBEDDINGS_TABLE}")
print(f"Endpoint: {EMBEDDING_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import requests
import base64
import io
import time
import numpy as np
from PIL import Image
from datetime import datetime
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType,
    DoubleType, TimestampType, BooleanType, IntegerType
)
from pyspark.sql import functions as F

# Get Databricks authentication
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
endpoint_url = f"{host}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations"

print(f"Endpoint URL: {endpoint_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Test Endpoint Connectivity

# COMMAND ----------

def test_endpoint():
    """Test that the FashionCLIP endpoint is responding."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Create a tiny test image (1x1 white pixel)
    img = Image.new('RGB', (64, 64), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    test_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    payload = {
        "dataframe_records": [{"image": test_b64, "text": ""}]
    }

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            if "predictions" in result:
                embedding = result["predictions"][0]
                # Handle different response formats
                if isinstance(embedding, dict):
                    dim = len(embedding.get("embedding", embedding.get("image_embedding", [])))
                elif isinstance(embedding, list):
                    dim = len(embedding)
                else:
                    dim = 0
                print(f"Endpoint test: SUCCESS")
                print(f"Embedding dimension: {dim}")
                return True
        else:
            print(f"Endpoint test: FAILED")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"Endpoint test: FAILED")
        print(f"Error: {e}")
        return False

# Run test
endpoint_ok = test_endpoint()
if not endpoint_ok:
    raise Exception("FashionCLIP endpoint is not responding. Check endpoint status.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Products

# COMMAND ----------

products_df = spark.table(PRODUCTS_TABLE)
total_products = products_df.count()

print(f"Total products: {total_products}")
display(products_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Define Embedding Functions

# COMMAND ----------

def read_image_as_base64(image_path):
    """Read image from path and convert to base64 string."""
    try:
        # Read image bytes from UC Volume
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Open with PIL to verify it's valid
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize if too large (max 512x512 for efficiency)
        max_size = 512
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert back to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None


def parse_embedding(prediction):
    """Parse embedding from different endpoint response formats."""
    if isinstance(prediction, dict):
        # Try different key names
        for key in ["embedding", "image_embedding", "image"]:
            if key in prediction:
                return prediction[key]
        # Return first list-like value
        for val in prediction.values():
            if isinstance(val, list):
                return val
    elif isinstance(prediction, list):
        return prediction
    return None


def generate_embeddings_batch(image_paths, product_ids):
    """Generate embeddings for a batch of images."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Prepare batch payload
    records = []
    valid_indices = []

    for i, (path, pid) in enumerate(zip(image_paths, product_ids)):
        b64 = read_image_as_base64(path)
        if b64:
            records.append({"image": b64, "text": ""})
            valid_indices.append(i)

    if not records:
        return [], []

    payload = {"dataframe_records": records}

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])

            embeddings = []
            for pred in predictions:
                emb = parse_embedding(pred)
                embeddings.append(emb if emb else [0.0] * 512)

            return embeddings, valid_indices
        else:
            print(f"Batch error: {response.status_code} - {response.text[:200]}")
            return [], []

    except Exception as e:
        print(f"Batch exception: {e}")
        return [], []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Embeddings for All Products

# COMMAND ----------

# Collect products to driver for processing
# (For larger datasets, consider distributed processing with UDFs)
products_list = products_df.select(
    "product_id", "display_name", "category", "sub_category",
    "color", "gender", "image_path"
).collect()

print(f"Processing {len(products_list)} products in batches of {BATCH_SIZE}")

# COMMAND ----------

# Process in batches
results = []
errors = []
start_time = time.time()

for batch_start in range(0, len(products_list), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(products_list))
    batch = products_list[batch_start:batch_end]

    image_paths = [row.image_path for row in batch]
    product_ids = [row.product_id for row in batch]

    # Generate embeddings
    embeddings, valid_indices = generate_embeddings_batch(image_paths, product_ids)

    # Store results
    for i, emb in zip(valid_indices, embeddings):
        row = batch[i]
        results.append({
            "product_id": row.product_id,
            "display_name": row.display_name,
            "category": row.category,
            "sub_category": row.sub_category,
            "color": row.color,
            "gender": row.gender,
            "image_path": row.image_path,
            "embedding": emb,
            "embedding_model": EMBEDDING_ENDPOINT,
            "embedding_dimension": len(emb),
            "is_valid": len(emb) == 512,
            "created_at": datetime.now()
        })

    # Track errors
    all_indices = set(range(len(batch)))
    failed_indices = all_indices - set(valid_indices)
    for i in failed_indices:
        errors.append(batch[i].product_id)

    # Progress update
    pct = (batch_end / len(products_list)) * 100
    print(f"Progress: {batch_end}/{len(products_list)} ({pct:.1f}%)")

    # Small delay to avoid rate limiting
    time.sleep(0.5)

elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.1f} seconds")
print(f"Successful: {len(results)}, Failed: {len(errors)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Save Embeddings to Delta Table

# COMMAND ----------

# Define schema for embeddings table
embedding_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("display_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("sub_category", StringType(), True),
    StructField("color", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("image_path", StringType(), True),
    StructField("embedding", ArrayType(DoubleType()), False),
    StructField("embedding_model", StringType(), True),
    StructField("embedding_dimension", IntegerType(), True),
    StructField("is_valid", BooleanType(), True),
    StructField("created_at", TimestampType(), True)
])

# Create DataFrame
embeddings_df = spark.createDataFrame(results, schema=embedding_schema)

# COMMAND ----------

# Preview embeddings
display(embeddings_df.select(
    "product_id", "display_name", "category", "embedding_dimension"
).limit(10))

# COMMAND ----------

# Check embedding validity
valid_count = embeddings_df.filter(F.col("is_valid") == True).count()
print(f"Valid embeddings (512-dim): {valid_count}/{len(results)}")

# Show embedding sample
sample_emb = embeddings_df.select("embedding").first()[0]
print(f"\nSample embedding (first 10 values):")
print(sample_emb[:10])

# COMMAND ----------

# Save to Delta table
embeddings_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(EMBEDDINGS_TABLE)

print(f"Saved embeddings to: {EMBEDDINGS_TABLE}")

# COMMAND ----------

# Enable Change Data Feed (required for Vector Search Delta Sync)
spark.sql(f"""
    ALTER TABLE {EMBEDDINGS_TABLE}
    SET TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
""")

print("Enabled Change Data Feed for Vector Search sync")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

final_count = spark.table(EMBEDDINGS_TABLE).count()

print("=" * 60)
print("EMBEDDING GENERATION COMPLETE")
print("=" * 60)
print(f"\nEmbedding endpoint: {EMBEDDING_ENDPOINT}")
print(f"Embeddings table: {EMBEDDINGS_TABLE}")
print(f"Total embeddings: {final_count}")
print(f"Embedding dimension: 512")
print(f"Processing time: {elapsed:.1f} seconds")
if errors:
    print(f"\nFailed products: {len(errors)}")
    print(f"Failed IDs: {errors[:10]}...")
print("\nNext step: Run 03_similarity_search.py")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC **"Endpoint not responding"**
# MAGIC - Check that `fashionclip-endpoint` is deployed and in "Ready" state
# MAGIC - Verify you have permission to invoke the endpoint
# MAGIC
# MAGIC **"Error reading image"**
# MAGIC - Ensure images exist in the UC Volume
# MAGIC - Check image paths in the products table
# MAGIC
# MAGIC **"Embedding dimension mismatch"**
# MAGIC - FashionCLIP should return 512-dim embeddings
# MAGIC - Check the endpoint model configuration
# MAGIC
# MAGIC **Slow processing**
# MAGIC - Increase BATCH_SIZE (max ~64 depending on image sizes)
# MAGIC - Use a cluster with more memory for larger batches
