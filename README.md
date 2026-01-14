# FashionCLIP Quickstart

Build a visual fashion search engine in 30 minutes using FashionCLIP embeddings and Databricks Vector Search.

**What you'll build:**
- Load a product catalog with images into Unity Catalog
- Generate 512-dimensional visual embeddings using FashionCLIP
- Create a vector index for similarity search
- Query: "Find products that look like this"

## Prerequisites

- **Databricks workspace** with Unity Catalog enabled
- **FashionCLIP endpoint** deployed and running (see [Endpoint Setup](#fashionclip-endpoint-setup) below)
- **Vector Search** enabled on your workspace
- Basic familiarity with Databricks notebooks

**No GPU required** - the FashionCLIP endpoint handles inference.

## Quick Start

### 1. Clone this repo to Databricks

```bash
# In your Databricks workspace
Repos > Add Repo > https://github.com/your-org/fashionclip-quickstart
```

### 2. Update configuration

Edit `config/config.yaml` with your Unity Catalog names:

```yaml
catalog: main
schema: fashion_quickstart
embedding_endpoint: fashionclip-endpoint
```

### 3. Run the notebooks in order

| Notebook | Purpose | Time |
|----------|---------|------|
| `01_load_data.py` | Load sample products to Unity Catalog | 2 min |
| `02_generate_embeddings.py` | Generate FashionCLIP embeddings | 5 min |
| `03_similarity_search.py` | Create vector index + run queries | 10 min |

### 4. Try a similarity query

After running all notebooks, you'll be able to search like this:

```python
# Find products similar to a given product
results = index.similarity_search(
    query_vector=product_embedding,
    columns=["product_id", "display_name", "category"],
    num_results=10
)
```

## Project Structure

```
fashionclip-quickstart/
├── README.md                 # This file
├── config/
│   └── config.yaml          # Configuration (edit this)
├── notebooks/
│   ├── 01_load_data.py      # Load products to Unity Catalog
│   ├── 02_generate_embeddings.py  # Generate FashionCLIP embeddings
│   └── 03_similarity_search.py    # Vector search setup + queries
├── data/
│   ├── products.csv         # 100 sample products
│   ├── images/              # Sample product images (add your own)
│   └── README.md            # Data sources + Kaggle instructions
└── requirements.txt         # Python dependencies
```

## Data Options

### Option 1: Use bundled sample data (easiest)
The repo includes `data/products.csv` with 100 products. Add your own images to `data/images/` or use placeholder images for testing.

### Option 2: Use Kaggle Fashion Dataset (recommended for real testing)
Download the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle:
1. Download and extract to your Databricks workspace
2. Set `DATA_SOURCE = "kaggle"` in notebook 01
3. Update `KAGGLE_PATH` to point to the extracted files

See `data/README.md` for detailed instructions.

## FashionCLIP Endpoint Setup

If you don't have a FashionCLIP endpoint yet:

### Option A: Use an existing CLIP endpoint
Any CLIP-based model serving endpoint that accepts base64 images and returns 512-dim embeddings will work. Update `embedding_endpoint` in config.

### Option B: Deploy FashionCLIP
1. Navigate to **Machine Learning > Serving** in Databricks
2. Create a new serving endpoint
3. Use the FashionCLIP model from Hugging Face: `patrickjohncyh/fashion-clip`
4. Configure input format to accept `{"image": "<base64>", "text": ""}`
5. Wait for endpoint to become "Ready" (~5 min)

## Sample Queries

### Find similar products by image
```python
# Get embedding for a product
product_emb = spark.table("main.fashion_quickstart.product_embeddings") \
    .filter("product_id = 'P001'").first().embedding

# Search for similar
results = index.similarity_search(
    query_vector=product_emb,
    num_results=10
)
```

### Filter by category
```python
# Only search in "Footwear"
results = index.similarity_search(
    query_vector=product_emb,
    filters={"category": "Footwear"},
    num_results=5
)
```

### Text-to-image search
```python
# Get text embedding from FashionCLIP
text_emb = get_text_embedding("blue denim jacket")

# Find matching products
results = index.similarity_search(
    query_vector=text_emb,
    num_results=10
)
```

## Customization

### Bring your own product catalog
1. Create a CSV with columns: `product_id`, `display_name`, `category`, `sub_category`, `color`, `gender`, `image_filename`
2. Upload product images to a Unity Catalog volume
3. Update the data loading in notebook 01

### Change embedding model
Update `embedding_endpoint` in config to use a different CLIP variant:
- Standard CLIP: General purpose
- FashionCLIP: Optimized for fashion
- SigLIP: Better performance on detailed images

### Scale to larger catalogs
- Increase `BATCH_SIZE` in notebook 02 for faster processing
- Use `CONTINUOUS` sync for real-time index updates
- Consider partitioning the embeddings table by category

## Next Steps

After completing this quickstart:

1. **Add more products** - Scale to your full catalog
2. **Build an application** - Use the Vector Search SDK in your app
3. **Explore the full solution** - Check out [visual-outfit-intelligence](https://github.com/your-org/visual-outfit-intelligence) for:
   - SAM segmentation for lookbook images
   - Outfit recommendations
   - Style clustering

## Troubleshooting

### "Endpoint not responding"
- Check that `fashionclip-endpoint` is in "Ready" state
- Verify you have invoke permissions on the endpoint

### "Permission denied creating schema"
- You need CREATE SCHEMA permission on the catalog
- Ask your workspace admin or use a different catalog

### "Index sync taking too long"
- Initial sync takes 5-15 minutes for Delta Sync indexes
- Check index status in the Vector Search UI

### "No similar products found"
- Verify embeddings were generated (check embeddings table)
- Ensure embedding dimensions match (should be 512)

## License

Apache 2.0

## Resources

- [FashionCLIP Paper](https://arxiv.org/abs/2204.03972)
- [Databricks Vector Search Docs](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html)
