# Databricks notebook source
# MAGIC %md
# MAGIC # 03: Create Vector Search Index & Run Similarity Queries
# MAGIC
# MAGIC This notebook creates a Databricks Vector Search index and demonstrates visual similarity queries.
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Creates a Vector Search endpoint (if needed)
# MAGIC 2. Creates a Delta Sync index on the embeddings table
# MAGIC 3. Runs sample similarity queries
# MAGIC 4. Demonstrates text-based search using FashionCLIP text embeddings
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Completed notebook 02 (embeddings table exists)
# MAGIC - Vector Search enabled on your workspace
# MAGIC
# MAGIC **Runtime:** CPU cluster (GPU not required)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration - UPDATE THESE
CATALOG = "main"
SCHEMA = "fashion_quickstart"

# Vector Search settings
VS_ENDPOINT_NAME = "fashion-vs-quickstart"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.product_embeddings_index"

# Source table
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.product_embeddings"

# Index configuration
PRIMARY_KEY = "product_id"
EMBEDDING_COLUMN = "embedding"
EMBEDDING_DIMENSION = 512

print(f"Embeddings table: {EMBEDDINGS_TABLE}")
print(f"Vector Search endpoint: {VS_ENDPOINT_NAME}")
print(f"Index name: {VS_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import time
import requests
import base64
import io
from PIL import Image

# Initialize Vector Search client
vsc = VectorSearchClient(disable_notice=True)

print("Vector Search client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Vector Search Endpoint

# COMMAND ----------

def get_or_create_endpoint(endpoint_name):
    """Get existing endpoint or create a new one."""
    try:
        endpoint = vsc.get_endpoint(name=endpoint_name)
        state = endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')
        print(f"Endpoint '{endpoint_name}' exists (state: {state})")
        return endpoint
    except Exception as e:
        print(f"Creating new endpoint '{endpoint_name}'...")
        endpoint = vsc.create_endpoint(
            name=endpoint_name,
            endpoint_type="STANDARD"
        )
        print(f"Endpoint '{endpoint_name}' created")
        return endpoint


def wait_for_endpoint(endpoint_name, timeout_minutes=10):
    """Wait for endpoint to become online."""
    print(f"Waiting for endpoint '{endpoint_name}' to be ready...")
    start = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start < timeout_seconds:
        endpoint = vsc.get_endpoint(name=endpoint_name)
        state = endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')
        print(f"  State: {state}")

        if state == "ONLINE":
            print("Endpoint is ready!")
            return True
        elif state in ["PROVISIONING", "STARTING"]:
            time.sleep(30)
        else:
            print(f"Unexpected state: {state}")
            return False

    print("Timeout waiting for endpoint")
    return False

# COMMAND ----------

# Get or create endpoint
endpoint = get_or_create_endpoint(VS_ENDPOINT_NAME)

# Wait for it to be ready
state = endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')
if state != "ONLINE":
    wait_for_endpoint(VS_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Delta Sync Index

# COMMAND ----------

def get_or_create_index(endpoint_name, index_name, source_table, primary_key, embedding_col, embedding_dim):
    """Get existing index or create a new one."""
    try:
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        state = index.get('status', {}).get('detailed_state', 'UNKNOWN')
        print(f"Index '{index_name}' exists (state: {state})")
        return index
    except Exception as e:
        print(f"Creating new index '{index_name}'...")
        index = vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",  # Manual sync trigger
            primary_key=primary_key,
            embedding_dimension=embedding_dim,
            embedding_vector_column=embedding_col
        )
        print(f"Index '{index_name}' created")
        return index


def wait_for_index(endpoint_name, index_name, timeout_minutes=15):
    """Wait for index to be ready."""
    print(f"Waiting for index '{index_name}' to be ready...")
    start = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start < timeout_seconds:
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        state = index.get('status', {}).get('detailed_state', 'UNKNOWN')
        print(f"  State: {state}")

        if state in ["ONLINE", "ONLINE_TRIGGERED_INDEXING"]:
            print("Index is ready!")
            return True
        elif state in ["PROVISIONING", "INDEXING", "PENDING", "ONLINE_NO_PENDING_UPDATE"]:
            time.sleep(30)
        else:
            print(f"State: {state}")
            time.sleep(30)

    print("Timeout waiting for index")
    return False

# COMMAND ----------

# Create or get index
index = get_or_create_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
    source_table=EMBEDDINGS_TABLE,
    primary_key=PRIMARY_KEY,
    embedding_col=EMBEDDING_COLUMN,
    embedding_dim=EMBEDDING_DIMENSION
)

# Wait for index to be ready
wait_for_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)

# COMMAND ----------

# Trigger sync (for TRIGGERED pipeline type)
try:
    vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME).sync()
    print("Index sync triggered")
except Exception as e:
    print(f"Note: {e}")
    print("(Index may already be syncing)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Index

# COMMAND ----------

# Get index details
index_info = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)

print("=" * 60)
print("INDEX INFORMATION")
print("=" * 60)
print(f"Name: {index_info.get('name')}")
print(f"Status: {index_info.get('status', {}).get('detailed_state')}")
print(f"Primary key: {PRIMARY_KEY}")
print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
print(f"Source table: {EMBEDDINGS_TABLE}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Run Similarity Search Queries
# MAGIC
# MAGIC ### Query Type 1: Find Similar Products by Product ID

# COMMAND ----------

# Get a sample product to use as query
sample_product = spark.table(EMBEDDINGS_TABLE).limit(1).collect()[0]
query_product_id = sample_product.product_id
query_product_name = sample_product.display_name
query_embedding = sample_product.embedding

print(f"Query product: {query_product_id} - {query_product_name}")
print(f"Category: {sample_product.category} / {sample_product.sub_category}")
print(f"Color: {sample_product.color}")

# COMMAND ----------

# Search for similar products
results = vsc.get_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME
).similarity_search(
    query_vector=query_embedding,
    columns=[PRIMARY_KEY, "display_name", "category", "sub_category", "color", "gender"],
    num_results=10
)

print(f"\nTop 10 Similar Products to '{query_product_name}':")
print("=" * 80)

if results and 'result' in results and 'data_array' in results['result']:
    for i, row in enumerate(results['result']['data_array'], 1):
        product_id = row[0]
        name = row[1]
        category = row[2]
        sub_cat = row[3]
        color = row[4]
        gender = row[5]
        score = row[-1]  # Similarity score is last

        # Skip the query product itself
        if product_id == query_product_id:
            continue

        print(f"{i}. {name}")
        print(f"   ID: {product_id} | {category}/{sub_cat} | {color} | {gender}")
        print(f"   Similarity: {score:.4f}")
        print()
else:
    print("No results returned. Index may still be syncing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Type 2: Text-Based Search
# MAGIC
# MAGIC FashionCLIP can also generate text embeddings, enabling text-to-image search.

# COMMAND ----------

# Get text embedding from FashionCLIP endpoint
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
endpoint_url = f"{host}/serving-endpoints/fashionclip-endpoint/invocations"

def get_text_embedding(text_query):
    """Get embedding for a text query using FashionCLIP."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Create a small placeholder image (FashionCLIP expects both image and text)
    img = Image.new('RGB', (64, 64), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Send text query with placeholder image
    payload = {
        "dataframe_records": [{"image": img_b64, "text": text_query}]
    }

    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        result = response.json()
        prediction = result["predictions"][0]

        # Try to get text embedding (may be in different keys)
        if isinstance(prediction, dict):
            return prediction.get("text_embedding", prediction.get("text", prediction.get("embedding")))
        return prediction
    else:
        print(f"Error: {response.status_code} - {response.text[:200]}")
        return None

# COMMAND ----------

# Search by text query
text_query = "blue denim jeans"
print(f"Text query: '{text_query}'")

text_embedding = get_text_embedding(text_query)

if text_embedding:
    text_results = vsc.get_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME
    ).similarity_search(
        query_vector=text_embedding,
        columns=[PRIMARY_KEY, "display_name", "category", "sub_category", "color"],
        num_results=10
    )

    print(f"\nProducts matching '{text_query}':")
    print("=" * 80)

    if text_results and 'result' in text_results and 'data_array' in text_results['result']:
        for i, row in enumerate(text_results['result']['data_array'], 1):
            name = row[1]
            category = row[2]
            sub_cat = row[3]
            color = row[4]
            score = row[-1]

            print(f"{i}. {name}")
            print(f"   {category}/{sub_cat} | {color}")
            print(f"   Similarity: {score:.4f}")
            print()
else:
    print("Could not generate text embedding. Endpoint may not support text queries.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Type 3: Filter + Search
# MAGIC
# MAGIC Combine vector similarity with metadata filters.

# COMMAND ----------

# Search for similar products, but only in "Footwear" category
filtered_results = vsc.get_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME
).similarity_search(
    query_vector=query_embedding,
    columns=[PRIMARY_KEY, "display_name", "category", "sub_category", "color"],
    filters={"category": "Footwear"},  # Only search footwear
    num_results=5
)

print(f"Similar products in 'Footwear' category:")
print("=" * 60)

if filtered_results and 'result' in filtered_results and 'data_array' in filtered_results['result']:
    for i, row in enumerate(filtered_results['result']['data_array'], 1):
        name = row[1]
        sub_cat = row[3]
        color = row[4]
        score = row[-1]

        print(f"{i}. {name}")
        print(f"   {sub_cat} | {color} | Similarity: {score:.4f}")
        print()
else:
    print("No footwear products found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Example API Usage
# MAGIC
# MAGIC Here's how to use the index from your application:

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC # Example: Search from your application
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC
# MAGIC # Initialize client (uses DATABRICKS_HOST and DATABRICKS_TOKEN from environment)
# MAGIC vsc = VectorSearchClient()
# MAGIC
# MAGIC # Get your index
# MAGIC index = vsc.get_index(
# MAGIC     endpoint_name="fashion-vs-quickstart",
# MAGIC     index_name="main.fashion_quickstart.product_embeddings_index"
# MAGIC )
# MAGIC
# MAGIC # Search by embedding vector (512-dim)
# MAGIC results = index.similarity_search(
# MAGIC     query_vector=your_embedding,  # 512-dim list
# MAGIC     columns=["product_id", "display_name", "category"],
# MAGIC     num_results=10
# MAGIC )
# MAGIC
# MAGIC # Parse results
# MAGIC for row in results['result']['data_array']:
# MAGIC     product_id = row[0]
# MAGIC     name = row[1]
# MAGIC     similarity = row[-1]
# MAGIC     print(f"{name}: {similarity:.3f}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("VECTOR SEARCH SETUP COMPLETE")
print("=" * 60)
print(f"\nEndpoint: {VS_ENDPOINT_NAME}")
print(f"Index: {VS_INDEX_NAME}")
print(f"Source table: {EMBEDDINGS_TABLE}")
print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
print(f"Products indexed: {spark.table(EMBEDDINGS_TABLE).count()}")
print("\nYou can now:")
print("  1. Search by image embedding (visual similarity)")
print("  2. Search by text (text-to-image)")
print("  3. Combine with metadata filters")
print("\nSee the API example above for integration.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC **"Endpoint not found"**
# MAGIC - Verify Vector Search is enabled on your workspace
# MAGIC - Check you have permissions to create VS endpoints
# MAGIC
# MAGIC **"Index sync pending"**
# MAGIC - Delta Sync indexes take 5-15 minutes to initially sync
# MAGIC - Run the sync trigger cell and wait
# MAGIC
# MAGIC **"No results returned"**
# MAGIC - Index may still be syncing
# MAGIC - Check that embeddings table has data
# MAGIC - Verify embedding dimension matches (512)
# MAGIC
# MAGIC **Text search not working**
# MAGIC - Not all CLIP endpoints support separate text embeddings
# MAGIC - Use image-based search as fallback
