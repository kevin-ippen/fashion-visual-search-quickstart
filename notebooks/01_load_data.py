# Databricks notebook source
# MAGIC %md
# MAGIC # 01: Load Sample Product Data
# MAGIC
# MAGIC This notebook loads sample product data into Unity Catalog for the FashionCLIP quickstart.
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Creates Unity Catalog schema and volume (if needed)
# MAGIC 2. Loads product metadata from CSV
# MAGIC 3. Uploads product images to UC Volume
# MAGIC 4. Creates the `products` Delta table
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Unity Catalog enabled workspace
# MAGIC - Permissions to create schemas/tables in your catalog
# MAGIC
# MAGIC **Runtime:** Any Databricks Runtime (CPU is fine)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Update these values to match your environment.

# COMMAND ----------

# Configuration - UPDATE THESE
CATALOG = "main"
SCHEMA = "fashion_quickstart"

# Data source options:
# "bundled" = use sample data from this repo
# "kaggle" = use downloaded Kaggle dataset
DATA_SOURCE = "bundled"

# If using Kaggle data, set this path (after downloading and extracting)
KAGGLE_PATH = "/Volumes/main/fashion_quickstart/kaggle_download/fashion-dataset"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Data source: {DATA_SOURCE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Schema and Volume

# COMMAND ----------

# Create schema if it doesn't exist
# NOTE: CREATE CATALOG requires admin permissions on most workspaces.
# If this fails, ask your workspace admin to grant access or use an existing catalog.
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Using: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# Create volume for product images
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.product_images
    COMMENT 'Product images for FashionCLIP embeddings'
""")

volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/product_images"
print(f"Volume path: {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Product Metadata

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType

# Define schema for products
product_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("display_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("sub_category", StringType(), True),
    StructField("color", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("season", StringType(), True),
    StructField("usage", StringType(), True),
    StructField("image_filename", StringType(), True)
])

# COMMAND ----------

if DATA_SOURCE == "bundled":
    # Load from repo's sample data
    # When running in Databricks, the repo is mounted at /Workspace/Repos/...
    # For local testing, adjust this path

    # Dynamically get current user for path resolution
    current_user = spark.sql("SELECT current_user()").first()[0]

    # Try multiple possible locations
    possible_paths = [
        f"/Workspace/Repos/{current_user}/fashion-visual-search-quickstart/data/products.csv",
        f"/Workspace/Users/{current_user}/fashion-visual-search-quickstart/data/products.csv",
        "/Workspace/Repos/fashion-visual-search-quickstart/data/products.csv",
        "file:/tmp/fashion-visual-search-quickstart/data/products.csv"
    ]

    products_df = None
    for path in possible_paths:
        try:
            products_df = spark.read.csv(path, header=True, schema=product_schema)
            print(f"Loaded products from: {path}")
            break
        except Exception as e:
            continue

    if products_df is None:
        # Create sample data inline as fallback
        print("Creating sample data inline...")
        sample_data = [
            ("P001", "Classic White Tee", "Apparel", "Tshirts", "White", "Men", "Summer", "Casual", "10001.jpg"),
            ("P002", "Black Skinny Jeans", "Apparel", "Jeans", "Black", "Men", "Fall", "Casual", "10002.jpg"),
            ("P003", "Navy Bomber Jacket", "Apparel", "Jackets", "Navy", "Men", "Fall", "Casual", "10003.jpg"),
            ("P004", "White Leather Sneakers", "Footwear", "Casual Shoes", "White", "Men", "Summer", "Casual", "10004.jpg"),
            ("P005", "Black Structured Handbag", "Accessories", "Handbags", "Black", "Women", "Fall", "Formal", "10005.jpg"),
        ]
        products_df = spark.createDataFrame(sample_data, schema=product_schema)

elif DATA_SOURCE == "kaggle":
    # Load from Kaggle Fashion Product Images Dataset
    # Expected structure: {KAGGLE_PATH}/styles.csv and {KAGGLE_PATH}/images/

    kaggle_csv = f"{KAGGLE_PATH}/styles.csv"
    print(f"Loading Kaggle data from: {kaggle_csv}")

    # Kaggle dataset has different column names - map them
    kaggle_df = spark.read.csv(kaggle_csv, header=True)

    products_df = kaggle_df.selectExpr(
        "CAST(id AS STRING) as product_id",
        "productDisplayName as display_name",
        "masterCategory as category",
        "subCategory as sub_category",
        "baseColour as color",
        "gender",
        "season",
        "usage",
        "CONCAT(CAST(id AS STRING), '.jpg') as image_filename"
    ).limit(100)  # Limit to 100 for quickstart

    print(f"Loaded {products_df.count()} products from Kaggle dataset")

# COMMAND ----------

# Preview data
display(products_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Upload Images to Volume
# MAGIC
# MAGIC For the bundled sample, we'll use placeholder images.
# MAGIC For Kaggle data, copy images from the downloaded dataset.

# COMMAND ----------

from pyspark.sql import functions as F

if DATA_SOURCE == "kaggle":
    # Copy images from Kaggle download to UC Volume
    kaggle_images_path = f"{KAGGLE_PATH}/images"

    # Get list of product IDs we're using
    product_ids = [row.product_id for row in products_df.select("product_id").collect()]

    # Copy only the images we need
    for pid in product_ids[:100]:  # Limit for quickstart
        src = f"{kaggle_images_path}/{pid}.jpg"
        dst = f"{volume_path}/{pid}.jpg"
        try:
            dbutils.fs.cp(src, dst)
        except Exception as e:
            print(f"Could not copy {pid}.jpg: {e}")

    print(f"Copied images to {volume_path}")

else:
    # For bundled data, images should already be in repo
    # Or create placeholder note
    print("Note: For bundled data, add your sample images to the UC Volume manually")
    print(f"Volume path: {volume_path}")
    print("Or download from Kaggle and run with DATA_SOURCE='kaggle'")

# COMMAND ----------

# Add full image path to products dataframe
products_with_path = products_df.withColumn(
    "image_path",
    F.concat(F.lit(volume_path + "/"), F.col("image_filename"))
)

display(products_with_path.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save to Delta Table

# COMMAND ----------

# Write products table
table_name = f"{CATALOG}.{SCHEMA}.products"

products_with_path.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"Created table: {table_name}")

# COMMAND ----------

# Verify
row_count = spark.table(table_name).count()
print(f"Products loaded: {row_count}")

# Show sample
display(spark.table(table_name).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What was created:**

# COMMAND ----------

print("=" * 60)
print("DATA LOADING COMPLETE")
print("=" * 60)
print(f"\nCatalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Products table: {table_name}")
print(f"Products loaded: {row_count}")
print(f"Image volume: {volume_path}")
print("\nNext step: Run 02_generate_embeddings.py")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC **"Cannot find products.csv"**
# MAGIC - Ensure the repo is properly cloned to your Databricks workspace
# MAGIC - Or use `DATA_SOURCE = "kaggle"` with downloaded Kaggle data
# MAGIC
# MAGIC **"Permission denied creating schema"**
# MAGIC - You need CREATE SCHEMA permission on the catalog
# MAGIC - Ask your workspace admin or use a different catalog
# MAGIC
# MAGIC **"No images in volume"**
# MAGIC - For Kaggle data: ensure you've downloaded and extracted the dataset
# MAGIC - For bundled data: manually upload sample images to the volume
