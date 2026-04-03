# Semantic Book Recommender

A semantic book recommendation project that starts with raw book metadata, enriches it with cleaner categories and emotion signals, and serves the final experience through a Gradio dashboard.

Instead of matching only exact keywords, this project uses embeddings to understand the meaning of a user query like:

`"A hopeful story about forgiveness"`  
`"A children's book about nature"`  
`"A suspenseful mystery with emotional depth"`

The result is a recommendation system that combines:

- Semantic search with LangChain + Chroma
- Category simplification and zero-shot labeling
- Emotion scoring from book descriptions
- A Gradio UI for interactive recommendations

## What This Project Does

The project builds a searchable book dataset in multiple stages:

1. Raw metadata is loaded from `books.csv`.
2. Incomplete rows are filtered out and text features are cleaned.
3. Descriptions are embedded into a vector database for semantic retrieval.
4. Missing categories are filled using a zero-shot classifier.
5. Emotion scores are extracted from book descriptions.
6. The final enriched dataset is saved and used by the Gradio dashboard.

The app can recommend books based on:

- Meaning of the user query
- Book category
- Emotional tone such as happy, sad, surprising, angry, or suspenseful

## Project Structure

- `data-exploration.ipynb`  
  Main notebook for data exploration, cleaning, semantic search experiments, category labeling, and emotion enrichment.

- `gradio-dashboard.py`  
  User-facing recommendation app built with Gradio.

- `gpu_zero_shot.py`  
  Helper for loading the zero-shot classification pipeline with the best available device: CUDA, Intel XPU, DirectML, or CPU.

- `books.csv`  
  Original dataset.

- `books_cleaned.csv`  
  Cleaned dataset after preprocessing and text feature preparation.

- `books_with_predicted_categories.csv`  
  Dataset after missing category labels are filled.

- `books_with_emotions.csv`  
  Final enriched dataset used by the dashboard.

## End-to-End Workflow

### 1. Data Acquisition and Setup

The notebook begins by downloading the source dataset with `kagglehub`, then loads the CSV into pandas for exploration.

Main tasks:

- Import analysis libraries like pandas, NumPy, seaborn, and matplotlib
- Load the raw books dataset
- Inspect the table before cleaning

### 2. Data Preprocessing and Cleaning

This stage improves data quality so later retrieval and classification steps work more reliably.

Main tasks:

- Visualize missing values with a heatmap
- Create helper features like:
  `missing_descriptioni`
  `age_of_book`
- Remove rows missing critical fields such as:
  `description`
  `average_rating`
  `num_pages`
  `published_year`
- Count words in each description
- Keep only books with at least 25 words in the description
- Merge `title` and `subtitle` into `title_and_subtitle`
- Build `tagged_description` by joining `isbn13` and `description`

Why `tagged_description` matters:

The semantic retriever later extracts the ISBN from retrieved text chunks, so including `isbn13` directly in the indexed text makes it easy to map vector search results back to the original book row.

### 3. Semantic Search with LangChain and Chroma

Once the text is cleaned, the notebook turns each description into an embedding-ready document and stores everything in a vector database.

Main tasks:

- Convert each `tagged_description` into a LangChain `Document`
- Split text with `CharacterTextSplitter`
- Embed documents with `HuggingFaceEmbeddings`
- Store embeddings in a Chroma vector database
- Test similarity search using natural-language queries

Example behavior:

A query like `"A book to teach children about nature"` retrieves books whose descriptions are semantically similar, even if they do not contain the exact same words.

### 4. Category Labeling with Zero-Shot Classification

The source categories are noisy and uneven, so the notebook simplifies them and fills missing category values.

Main tasks:

- Map detailed categories into simpler labels such as:
  `Fiction`
  `Nonfiction`
  `Children's Fiction`
  `Children's Nonfiction`
- Evaluate zero-shot predictions on known fiction and nonfiction samples
- Measure prediction accuracy on a small validation slice
- Predict categories for books where the simplified category is missing
- Merge the predicted labels back into the dataset

Why this step matters:

The Gradio dashboard lets users filter by category. That works best when category labels are consistent and missing values are reduced.

### 5. Emotion Tagging and Final Export

The final enrichment step adds emotional metadata extracted from each book description.

Main tasks:

- Load the model `j-hartmann/emotion-english-distilroberta-base`
- Split each description into sentences
- Score each sentence across:
  `anger`
  `disgust`
  `fear`
  `joy`
  `sadness`
  `surprise`
  `neutral`
- Keep the maximum score for each emotion across all sentences in the description
- Merge those emotion features into the books dataframe
- Save the final dataset as `books_with_emotions.csv`

Why max emotion scores are used:

A single strong emotional moment in a description can be meaningful for recommendation filtering. Taking the maximum score captures the strongest presence of each emotional tone.

## How the Gradio Dashboard Works

The dashboard in `gradio-dashboard.py` uses `books_with_emotions.csv` as its source of truth.

### Input controls

Users provide:

- A free-text query
- An optional category filter
- An optional emotional tone filter

### Retrieval flow

The app:

1. Loads the final enriched CSV.
2. Rebuilds `tagged_description` if needed.
3. Creates a Chroma vector store from all books.
4. Runs semantic similarity search for the user query.
5. Extracts ISBNs from retrieved chunks.
6. Reorders the matching books by semantic rank.
7. Applies category filtering if selected.
8. Re-sorts by emotion score if a tone is selected.
9. Returns the final recommendations in a Gradio gallery.

### Tone mapping used in the app

The UI tone filters map to emotion-score columns like this:

- `Happy` → `joy`
- `Surprising` → `surprise`
- `Angry` → `anger`
- `Suspenseful` → `fear`
- `Sad` → `sadness`

### Output format

Each recommendation shows:

- Book cover thumbnail
- Title
- Author name(s)
- A short truncated description

## Recommendation Logic

The project does not rely on one signal alone. Recommendations come from a combination of signals:

- Semantic similarity finds meaning-based matches
- Simplified categories keep results relevant to the user’s preferred type of book
- Emotion scores shape the mood of the final ranking

That makes the system more expressive than a plain keyword search engine.

## Running the Project

### 1. Create and activate an environment

Install dependencies in your preferred environment manager. This project uses Python libraries such as:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `python-dotenv`
- `gradio`
- `transformers`
- `torch`
- `langchain`
- `langchain-chroma`
- `langchain-huggingface`
- `langchain-text-splitters`
- `chromadb`
- `kagglehub`
- `tqdm`

### 2. Run the notebook

Open:

- `data-exploration.ipynb`

Run the notebook from top to bottom to reproduce the dataset-building pipeline.

### 3. Launch the dashboard

```powershell
python gradio-dashboard.py
```

Then open the local Gradio URL in your browser.

## Data Files Produced by the Notebook

- `books_cleaned.csv`  
  Cleaned records after preprocessing.

- `books_with_predicted_categories.csv`  
  Records after filling missing simplified categories.

- `books_with_emotions.csv`  
  Final dataset with semantic-search-ready text, category labels, and emotion scores.

## Models and Libraries Used

### Embedding and retrieval

- `HuggingFaceEmbeddings`
- `Chroma`
- `LangChain`

### Classification

- `facebook/bart-large-mnli` for zero-shot category labeling
- `j-hartmann/emotion-english-distilroberta-base` for emotion scoring

### Interface

- `Gradio`

## Notes

- The notebook is both exploratory and production-oriented, so it contains experiments as well as final export steps.
- The dashboard builds the vector store at runtime from the final CSV.
- `gpu_zero_shot.py` helps the zero-shot pipeline use GPU acceleration when available.
- If thumbnails are missing, the dashboard falls back to a placeholder image.

## Future Improvements

- Save the vector database to disk instead of rebuilding it at app startup
- Add hybrid retrieval with metadata-aware reranking
- Improve category taxonomy beyond fiction and nonfiction style grouping
- Cache model pipelines for faster repeated notebook experiments
- Add deployment instructions for Hugging Face Spaces or another hosting platform

## Summary

This project is a full semantic recommendation pipeline:

- raw metadata → cleaned dataset
- cleaned dataset → vector search + category enrichment
- enriched dataset → emotion-aware recommendations
- final dataset → interactive Gradio experience

It is a strong example of combining NLP, embeddings, classification, and UI development into one practical machine learning application.
