# Vibe Coding with Kev: Visual Search on Databricks in 30 Minutes

*First post in a series about learning by doing. A non-engineer's adventures building actual things.*

---

I'm an Enterprise Account Executive with a liberal arts degree. I sell Databricks. I don't use it. Or at least, I didn't.

Then I started wondering if the tools are finally good enough for someone like me to build something real. Not a tutorial. Not a quick Genie space. An end-to-end thing that solves a problem.

So I tried. Here's what happened.

---

## The Project

Goal was simple: visual search. Upload an image and get similar products. This came from a real customer ask. I've lived a lot of my career in eCommerce, so it felt like the right hill to charge.

The question: could I actually ship something useful without a data science background?

---

## What I Built

A pipeline that:

1. **Loads product data** into Unity Catalog with images in a managed Volume
2. **Generates embeddings** using FashionCLIP (a vision-language model tuned for fashion)
3. **Creates a vector index** for nearest-neighbor search
4. **Answers queries** like "find products that look like this" or "show me blue denim jackets"

Three notebooks. Maybe an hour of work. Visual search that actually works.

---

## The Data

I used the [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). 44K+ products with images, categories, colors, and descriptions. Good starter set for fashion search.

The repo includes a smaller sample (100 products) so you can run through the whole thing without downloading gigabytes.

---

## The Model

**FashionCLIP.** It's a CLIP-style model fine-tuned for fashion. It puts images and text into the same 512-dimensional vector space, so "flowy summer dress" can be compared directly to an actual product photo.

This is the heart of multimodal search. Same embeddings, same index, two query types.

I didn't train anything. Didn't fine-tune anything. Just called an endpoint and stored the vectors.

---

## The Platform

Databricks carried a lot of weight:

**Unity Catalog** for governance. The products table lives in a managed Delta table. Images sit in a UC Volume. Everything namespaced, everything tracked.

**Model Serving** to host FashionCLIP. I deployed the endpoint once, and it handles all inference. Scales to zero when idle. I get rate limits, usage tracking, and inference tables out of the box.

**Vector Search** built an index directly from my embeddings table. Delta Sync keeps it fresh. Same endpoint handles both image similarity and text search. I typed "red cocktail dress," got relevant products. No extra infrastructure.

**No GPU wrestling.** The ML runtime has what I need. The endpoint handles inference. I ran everything on a basic CPU cluster.

---

## The Notebooks

| Notebook | What it does | Time |
|----------|--------------|------|
| `01_load_data.py` | Creates schema, volume, loads product CSV | 2 min |
| `02_generate_embeddings.py` | Calls FashionCLIP, stores 512-dim vectors | 5 min |
| `03_similarity_search.py` | Creates Vector Search index, runs queries | 10 min |

That's it. The README walks through setup. If you have a Databricks workspace with Unity Catalog and Vector Search enabled, you can run this.

---

## What Works

**Visual similarity is legit.** Pick a product, find things that look like it. The embeddings capture more nuance than I expected from a model I didn't train. White sneakers return other white sneakers. Black leather jackets find other black leather jackets. It just works.

**Text queries work on the same index.** "Blue denim jeans" returns blue denim jeans. Same embeddings, same infrastructure. The multimodal magic comes free with CLIP-style models.

**Filtered search.** Combine vectors with metadata. "Find products similar to this, but only in Footwear." Vector Search handles the filter pushdown.

---

## What I Learned

**The barrier dropped.** I stitched together something that would have needed an ML team a few years ago. Not because I'm special, but because the models exist and the platform handles the infrastructure parts.

**Starting is easier than it used to be.** FashionCLIP is on HuggingFace. Model Serving deploys it. Vector Search indexes it. I connected pieces. I didn't build them.

**Databricks Assistant helped more than I expected.** When I got stuck on the Vector Search SDK, I asked. It gave me working code. Not perfect, but enough to unblock myself.

---

## Try It Yourself

If you're new to Databricks, this is a good first project:

1. **Clone the repo** to your Databricks workspace
2. **Deploy a FashionCLIP endpoint** (or use an existing CLIP endpoint)
3. **Update config.yaml** with your catalog/schema names
4. **Run the three notebooks in order**

You'll have working visual search in about 30 minutes.

**Repo:** [fashionclip-quickstart](https://github.com/your-org/fashionclip-quickstart)

---

## What's Next

This is Part 1. Visual search is useful on its own, but I couldn't stop there.

**Part 2** goes deeper: taking inspiration images (lookbooks, editorial shots), segmenting individual garments using SAM (Segment Anything Model), and building "complete the look" recommendations. That's messier. Outfit rules don't map cleanly to co-occurrence data. Accessories hijack recommendations. I'm still tuning it.

But that's the point. The whole exercise is learning by doing. If you've been AI-curious and assumed you needed more technical depth, try a small project and see how far you get.

The goalposts moved.

---

## Resources

- [FashionCLIP Paper](https://arxiv.org/abs/2204.03972)
- [Kaggle Fashion Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- [Databricks Vector Search Docs](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html)

---

*Questions? Want the full pipeline with DABs? Reach out.*
