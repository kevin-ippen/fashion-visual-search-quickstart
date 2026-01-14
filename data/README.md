# Sample Data

This directory contains sample product data for the FashionCLIP quickstart.

## Contents

### products.csv
100 sample products with metadata:
- `product_id` - Unique identifier (P001-P100)
- `display_name` - Product name
- `category` - Top-level category (Apparel, Footwear, Accessories)
- `sub_category` - Detailed category (Tshirts, Jeans, Sneakers, etc.)
- `color` - Primary color
- `gender` - Men, Women, or Unisex
- `season` - Summer, Fall, Winter, Spring
- `usage` - Casual, Formal, Sports, Party
- `image_filename` - Expected image filename (10001.jpg, etc.)

### images/
Add your product images here. Expected filenames match `image_filename` in products.csv.

For testing without images, notebook 01 will create placeholder entries.

## Data Sources

### Option 1: Bundled Sample (Included)
The `products.csv` file is ready to use. Just add corresponding images to `images/`.

### Option 2: Kaggle Fashion Product Images Dataset (Recommended)

For real product images, download from Kaggle:

**Dataset:** [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

**Steps:**
1. Create a Kaggle account (free)
2. Download the dataset (~25GB, or use the small version)
3. Extract to your Databricks workspace or local machine
4. Upload to a Unity Catalog volume:
   ```python
   # In Databricks
   dbutils.fs.cp(
       "file:/path/to/fashion-dataset",
       "/Volumes/main/fashion_quickstart/kaggle_download/fashion-dataset",
       recurse=True
   )
   ```
5. In notebook 01, set:
   ```python
   DATA_SOURCE = "kaggle"
   KAGGLE_PATH = "/Volumes/main/fashion_quickstart/kaggle_download/fashion-dataset"
   ```

**Kaggle Dataset Structure:**
```
fashion-dataset/
├── styles.csv           # Product metadata (44K+ products)
├── images/              # Product images
│   ├── 10001.jpg
│   ├── 10002.jpg
│   └── ...
└── images.csv           # Image metadata
```

### Option 3: Your Own Data

To use your own product catalog:

1. Create a CSV with these columns:
   ```
   product_id,display_name,category,sub_category,color,gender,image_filename
   ```

2. Place your images in a folder, named to match `image_filename`

3. Update notebook 01 to load from your paths

## Image Requirements

For best results with FashionCLIP:
- **Format:** JPG or PNG
- **Size:** At least 224x224 pixels (model resizes automatically)
- **Content:** Clear product shots on neutral backgrounds work best
- **Quality:** Higher quality images = better embeddings

## Sample Data Distribution

The bundled `products.csv` includes:

| Category | Count | Sub-categories |
|----------|-------|----------------|
| Apparel | 60 | Tshirts, Shirts, Jackets, Jeans, Trousers, Dresses |
| Footwear | 20 | Casual Shoes, Formal Shoes, Sneakers |
| Accessories | 20 | Handbags, Backpacks, Watches, Sunglasses, Belts, Caps |

Gender split: Men (50%), Women (45%), Unisex (5%)

## Notes

- The quickstart notebooks limit to 100 products for fast iteration
- For production, scale to your full catalog
- FashionCLIP works best on fashion/apparel images
- For non-fashion products, consider using standard CLIP instead
