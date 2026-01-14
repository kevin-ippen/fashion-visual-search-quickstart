# Vibe Coding with Kev, Part 2: Complete the Look with SAM and Databricks

*Second post in a series about learning by doing. Now we're getting into the messy stuff.*

---

In [Part 1](./BLOG_PART1.md), I built visual search in about 30 minutes. Upload an image, get similar products. Clean, simple, useful.

Then I couldn't help myself.

---

## The Ambitious Part

Visual search is great, but e-commerce brands want more. They want "complete the look." Show someone an inspiration photo from a lookbook and surface the products that match. Even better, show items that pair well together. Build an outfit, not just find a single product.

So I went down that road. Here's what I learned.

---

## What I Tried to Build

A pipeline that:

1. **Takes inspiration images** (lookbooks, editorial shots, styled outfits)
2. **Segments individual pieces** (shirts, pants, bags, shoes) using SAM
3. **Generates embeddings** for each segmented region
4. **Matches those regions** to products in a catalog
5. **Recommends items** that appear together in outfits

The architecture looked clean on paper:

```
Lookbook Images
      ↓
SAM Segmentation → Individual Items (~50 regions per image)
      ↓
FashionCLIP Embeddings (same 512-dim vectors)
      ↓
Similarity Search ↔ Product Catalog
      ↓
Co-occurrence Analysis → "Complete the Look" Recommendations
```

---

## The Good Parts

**SAM (Segment Anything Model) is impressive.** Meta's zero-shot segmentation model handles fashion images surprisingly well. I used grid-based prompting (8x8 points across each image) and let SAM generate candidate masks. After filtering by area and confidence, I got 30-70 usable regions per lookbook image.

**The same FashionCLIP endpoint worked for everything.** Cropped regions from lookbooks got the same 512-dim embeddings as catalog products. Same endpoint, same vectors, same similarity math.

**Bidirectional search opened up new use cases.** Not just "find products that look like this region," but also "find lookbook images where this product would fit." Useful for merchandising and content curation.

**Co-occurrence actually surfaced patterns.** Products that appear together in styled lookbooks tend to pair well. The math is simple: group products by lookbook, generate pairs, count how often each pair appears together. More counts = stronger pairing signal.

---

## The Messy Parts

**Accessories hijack everything.** Watches, belts, and sunglasses appear in tons of lookbooks but don't really "pair" with anything specific. They're visually present without being outfit-dependent. My raw recommendations were dominated by accessories with artificially high co-occurrence counts.

The fix: statistical outlier filtering. Compute the 75th percentile co-occurrence count, add 2x the standard deviation, and filter products above that threshold. It's a blunt instrument, but it helped.

**Mixed outfits in single images.** Some lookbooks show multiple people or multiple outfit combinations. When you naively group by image, you create false pairings between items that were never meant to go together.

I don't have a clean solution yet. Better segmentation metadata could help. Or manual curation of which regions belong to which "outfit context" within an image. This is where domain expertise matters.

**SAM is slow on GPU.** Each image takes 30-60 seconds on a T4. Processing 100 lookbooks means 30-60 minutes of GPU time. Not a blocker, but it shapes how you think about refresh cadence.

**Similarity thresholds are arbitrary.** How similar is "similar enough"? I picked top-10 matches per region and top-5 regions per product. Those numbers felt right. They're not principled.

---

## The Platform Pieces

**GPU compute on demand.** SAM runs on an NC-series cluster (Tesla T4). I spin it up for segmentation and embedding generation, then shut it down. No CUDA driver pain. Just select the ML runtime and go.

**Model Serving for FashionCLIP.** Same endpoint from Part 1. Autoscales to zero when idle. Handles both product and region embeddings.

**Unity Catalog for everything.** Products, embeddings, regions, matches, recommendations. All in UC tables. Images in Volumes. Lineage tracked automatically.

**Databricks Asset Bundles for deployment.** One `databricks.yml` file defines the full pipeline: six notebooks, compute specs, job schedules. Deploy with `databricks bundle deploy`. Run with `databricks bundle run`. Promote dev to prod by changing the target.

---

## The Numbers

For a batch of ~100 lookbook images and ~44K products:

| Stage | Compute | Time |
|-------|---------|------|
| SAM Segmentation | GPU (T4) | 30-60 min |
| Quality Filtering | CPU | 2-5 min |
| Region Embeddings | GPU | 5-10 min |
| Similarity Search | CPU | 5-10 min |
| Co-occurrence Analysis | CPU | 2-5 min |
| **Total** | | **45-90 min** |

Output: ~2.8M product pairs with co-occurrence counts. After filtering, maybe 70% of the catalog has at least one recommendation.

---

## What I'd Do Differently

**Start with cleaner lookbook data.** Single-outfit images with clear item boundaries would make segmentation much more useful. Mixed-outfit photos create noise.

**Use Vector Search from the start.** I did batch similarity with Spark cross-joins. Fine for prototyping. For production, Databricks Vector Search would handle real-time queries and scale to larger catalogs.

**Build in human-in-the-loop curation.** Some recommendations are obviously wrong. A quick UI for thumbs-up/thumbs-down would create training signal for a better model later.

**Think harder about slot planning.** I built a simple SlotPlanner that maps seed items to complementary categories (jacket → pants + shoes + bag). The rules are hand-coded. A real system would learn these mappings from data.

---

## The Honest Assessment

**Visual search works.** Part 1 is a win. Ship it.

**Complete the look is a prototype.** The signal is there. Co-occurrence from lookbooks produces plausible recommendations. But the edge cases are sharp. Accessories, mixed outfits, arbitrary thresholds. It needs refinement before I'd put it in front of customers.

The jump from "demo" to "production" is real. I learned a lot. I have something functional. I'm not done.

---

## Try It Yourself

The full solution is in the [visual-outfit-intelligence](https://github.com/your-org/visual-outfit-intelligence) repo:

- **6 notebooks** covering the full pipeline
- **Slot planning logic** for outfit completion
- **DAB configuration** for one-command deployment
- **Sample data** so you can run it without sourcing your own lookbooks

Prerequisites:
- Databricks workspace with Unity Catalog
- GPU cluster access (Standard_NC6s_v3 or similar)
- FashionCLIP endpoint deployed

---

## What's Next

Part 3 (if I get there) will explore:

- **Fine-tuning** with feedback data
- **Vector Search** for real-time inference
- **Style clustering** to group lookbooks by aesthetic
- **A simple UI** for trying this end-to-end

Or maybe I'll pivot to something else entirely. That's the point of learning by doing. Follow the interesting threads.

---

## Resources

- [SAM (Segment Anything) Paper](https://arxiv.org/abs/2304.02643)
- [FashionCLIP Paper](https://arxiv.org/abs/2204.03972)
- [Kaggle Fashion Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Databricks Asset Bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html)

---

*Questions? Want to compare notes on outfit recommendation approaches? Reach out.*
