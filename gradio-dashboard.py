from __future__ import annotations

from typing import Optional

import gradio as gr
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

BOOKS_CSV_PATH = "books_with_emotions.csv"
FALLBACK_THUMBNAIL = "https://placehold.co/320x480?text=No+Cover"
EMOTION_COLUMN_BY_TONE = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


def load_books() -> pd.DataFrame:
    books = pd.read_csv(BOOKS_CSV_PATH).copy()

    if "tagged_description" not in books.columns:
        books["tagged_description"] = books[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)

    books["large_thumbnail"] = books["thumbnail"].fillna("").map(build_thumbnail_url)
    return books


def build_thumbnail_url(thumbnail: str) -> str:
    thumbnail = thumbnail.strip()
    if not thumbnail:
        return FALLBACK_THUMBNAIL

    if "fife=w800" in thumbnail:
        return thumbnail

    separator = "&" if "?" in thumbnail else "?"
    return f"{thumbnail}{separator}fife=w800"


def build_vector_store(books: pd.DataFrame) -> Chroma:
    raw_documents = [Document(page_content=row) for row in books["tagged_description"]]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_documents(documents, embedding=embeddings)


def extract_isbn13(document: Document) -> Optional[int]:
    parts = document.page_content.strip('"').split()
    if not parts:
        return None

    try:
        return int(parts[0])
    except ValueError:
        return None


def format_authors(authors: object) -> str:
    if pd.isna(authors):
        return "Unknown author"

    names = [name.strip() for name in str(authors).split(";") if name.strip()]
    if not names:
        return "Unknown author"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def truncate_description(description: object, word_limit: int = 30) -> str:
    if pd.isna(description):
        return "No description available."

    words = str(description).split()
    if len(words) <= word_limit:
        return " ".join(words)
    return " ".join(words[:word_limit]) + " ..."


BOOKS = load_books()
DB_BOOKS: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    global DB_BOOKS
    if DB_BOOKS is None:
        DB_BOOKS = build_vector_store(BOOKS)
    return DB_BOOKS


def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    clean_query = query.strip()
    if not clean_query:
        return BOOKS.iloc[0:0].copy()

    recs = get_vector_store().similarity_search(clean_query, k=initial_top_k)

    ordered_isbns: list[int] = []
    seen_isbns: set[int] = set()
    for rec in recs:
        isbn = extract_isbn13(rec)
        if isbn is None or isbn in seen_isbns:
            continue
        seen_isbns.add(isbn)
        ordered_isbns.append(isbn)

    if not ordered_isbns:
        return BOOKS.iloc[0:0].copy()

    ranking = {isbn: index for index, isbn in enumerate(ordered_isbns)}
    book_recs = BOOKS[BOOKS["isbn13"].isin(ordered_isbns)].copy()
    book_recs["semantic_rank"] = book_recs["isbn13"].map(ranking)
    book_recs.sort_values("semantic_rank", inplace=True)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].copy()

    tone_column = EMOTION_COLUMN_BY_TONE.get(tone)
    if tone_column:
        book_recs.sort_values(by=tone_column, ascending=False, inplace=True)

    return book_recs.head(final_top_k).drop(columns="semantic_rank", errors="ignore")


def recommend_books(query: str, category: str, tone: str) -> list[tuple[str, str]]:
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results: list[tuple[str, str]] = []

    for _, row in recommendations.iterrows():
        authors_str = format_authors(row.get("authors"))
        truncated_description = truncate_description(row.get("description"))
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


categories = ["All"] + sorted(BOOKS["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All",
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All",
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2, object_fit="contain")

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )


if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Glass())
