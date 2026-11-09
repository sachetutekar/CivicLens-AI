# CivicLens AI — Research Intelligence Studio

Interactive Gradio app using Gemini and in-memory hybrid retrieval (TF-IDF word/character + LSA + reciprocal rank fusion).

## Local run

1. Install requirements: `pip install -r requirements.txt`
2. Set `GEMINI_API_KEY` as an environment variable (do not commit it).
3. Run `python app.py`, then visit `http://localhost:7860`.

## Render deployment

1. Push these files to a GitHub repository.
2. Create a Render Web Service from the repository, or use `render.yaml` as a blueprint.
3. Build command: `pip install -r requirements.txt`; start command: `python app.py`.
4. Add `GEMINI_API_KEY` to Render Environment settings. Use Free instance for testing.
5. Deploy; test uploads, indexing, answering, auditing, and downloads.

## Important caveats

- Free Render services sleep when idle and have ephemeral files.
- Each visitor now has a distinct *in-memory* document index (Gradio session state); documents/indexes are lost on restart/session expiry.
- Uploaded document files and brief outputs are temporarily cached server-side by Gradio. Do not upload private/confidential documents.
- The public site consumes *your* Gemini API quota; use API restrictions/budgets and consider authentication or rate limiting before broad sharing.
- Large PDFs and indexing may exceed free-service RAM and CPU capacity; test a small public document first.
