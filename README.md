# jina-embeddings-v5-omni-small-classification demo

Streamlit demo of a single 1024-d vector space shared by **text, image, video and audio**. Embeddings come from
Elastic Inference Service (`.jina-embeddings-v5-omni-small`) and are stored/searched in Elasticsearch (`dense_vector`, cosine kNN).
Index with text, query with any modality; no reindexing.

## Run
```bash
pip install -r requirements.txt     # ffmpeg must be on PATH (video generation)
cp .env.example .env                # set ELASTICSEARCH_URL / ELASTICSEARCH_API_KEY
streamlit run app.py
```
The API key needs index read/write and inference privileges.

## Tabs
1. **Datasets & index** – generates synthetic samples (8 images, 6 videos, 8 audio clips, 30 text docs: a caption per
   media item plus unrelated notes), embeds each with its own modality via EIS, and bulk-indexes them.
2. **Cross-modal search** – query with text, an image/video/audio sample or your own upload; top-k per target modality.
3. **Zero-shot classification** – labels are embedded as text; any modality is assigned to its nearest label (softmax over cosine). Matryoshka dims selectable.
4. **Shared-space check** – cosine between text embedded by the omni endpoint and the text-only endpoint (≈1.0 in testing).

## EIS request shape
`POST _inference/embedding/.jina-embeddings-v5-omni-small`
```json
{"input": [{"content": [{"type": "image", "format": "base64", "value": "data:image/png;base64,..."}]}]}
```
Types: `text | image | audio | video | pdf`. Media must be a `data:` URI; max 16 inputs per request. Text gets the
`Document: ` prefix the model card requires (toggle in the sidebar).

## Notes
- Cross-modal cosine scores are lower than same-modality ones, hence per-modality ranking in search.
- Synthetic audio (tones, noise) maps to captions less cleanly than shapes do; swap in real files via the upload widgets.

## License

- **Model:** [`jinaai/jina-embeddings-v5-omni-small-classification`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-classification)
  is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): attribution required, **non-commercial use only**.
  For commercial use, contact Jina AI (sales@jina.ai). Using it through Elastic Inference Service does not change the model's license terms.
- **Sample data:** generated procedurally by `datasets.py`; no third-party content is included.
- **Demo code:** no license has been chosen for this repository yet. Add a `LICENSE` file before redistributing it.
