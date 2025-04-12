import os
import re
import json
import time
import argparse
import hashlib
import logging
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
import pysbd  # pip install pysbd

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("resume_parser.log"), logging.StreamHandler()]
)

# -----------------------------
# Custom Sentence Splitting with pysbd
# -----------------------------
def split_sentences_with_pysbd(text):
    """Split text into meaningful sentences using pysbd with custom merging."""
    segmenter = pysbd.Segmenter(language="en", clean=False)
    raw_sentences = segmenter.segment(text)
    meaningful = []
    buffer = ""
    for sent in raw_sentences:
        # Clean unwanted bullet characters and extra spaces.
        sent_clean = re.sub(r"[\u2022\u25AA\u25CF\u25A0•▪➢]", " ", sent)
        sent_clean = re.sub(r"\s+", " ", sent_clean).strip()
        if len(sent_clean.split()) < 5:
            buffer += " " + sent_clean
        else:
            if buffer:
                sent_clean = (buffer.strip() + " " + sent_clean).strip()
                buffer = ""
            meaningful.append(sent_clean)
    if buffer:
        meaningful.append(buffer.strip())
    return meaningful

# -----------------------------
# Resume Processing & Metadata Generation
# -----------------------------
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    except Exception as e:
        logging.error(f"Error computing hash for {file_path}: {e}")
    return hasher.hexdigest()

def compute_sentence_hash(sentence):
    return hashlib.sha1(sentence.encode("utf-8")).hexdigest()

def extract_clean_text(file_path, tika_url="http://localhost:9998/rmeta/text"):
    """Extract text using Apache Tika."""
    try:
        with open(file_path, "rb") as f:
            headers = {"Accept": "application/json"}
            response = requests.put(tika_url, data=f, headers=headers)
            response.raise_for_status()
            content_json = response.json()
            if content_json and isinstance(content_json, list):
                return content_json[0].get("X-TIKA:content", "").strip()
    except Exception as e:
        logging.error(f"❌ Error extracting from {file_path}: {e}")
    return ""

def process_file(file, pdf_dir, model, tika_url, existing_sentence_hashes):
    file_path = os.path.join(pdf_dir, file)
    file_hash = compute_file_hash(file_path)
    text = extract_clean_text(file_path, tika_url)
    if not text:
        return []
    sentences = split_sentences_with_pysbd(text)
    new_sentences = []
    results = []
    for sentence in sentences:
        if len(sentence.split()) < 5:
            continue
        sent_hash = compute_sentence_hash(sentence)
        if sent_hash in existing_sentence_hashes:
            continue
        new_sentences.append(sentence)
        results.append({
            "file": file,
            "file_hash": file_hash,
            "sentence": sentence,
            "length": len(sentence),
            "sentence_hash": sent_hash
        })
    if new_sentences:
        passages = [f"passage: {s}" for s in new_sentences]
        # Here progress bar is enabled during build.
        embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
        for idx, emb in enumerate(embeddings):
            results[idx]["embedding"] = emb.tolist()
    else:
        results = []
    return results

def load_existing_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def process_all_files(pdf_dir, model, tika_url, metadata_file):
    supported_exts = ('.pdf', '.doc', '.docx', '.pptx', '.txt', '.rtf')
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(supported_exts)]
    if not files:
        logging.info("📂 No supported files found.")
        return []
    existing_metadata = load_existing_metadata(metadata_file)
    processed_sentence_hashes = {entry.get("sentence_hash") for entry in existing_metadata if entry.get("sentence_hash")}
    new_results = []
    # Use a progress bar to track file processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, pdf_dir, model, tika_url, processed_sentence_hashes) for file in files]
        for future in tqdm(futures, desc="🔁 Processing Files", total=len(futures)):
            file_results = future.result()
            if file_results:
                new_results.extend(file_results)
    return new_results

def build_incremental_index(new_results, model, index_file, metadata_file):
    existing_metadata = load_existing_metadata(metadata_file)
    all_metadata = existing_metadata.copy()
    sample_vec = model.encode("example", normalize_embeddings=True, show_progress_bar=False)
    dim = sample_vec.shape[0]
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatIP(dim)
    new_vectors = []
    for item in new_results:
        vec = np.array(item["embedding"]).astype("float32")
        new_vectors.append(vec)
        item.pop("embedding", None)
        all_metadata.append(item)
    if new_vectors:
        new_vectors = np.array(new_vectors)
        # Show a progress bar while adding vectors.
        for vec in tqdm(new_vectors, desc="📊 Adding Vectors to Index"):
            index.add(vec.reshape(1, -1))
        faiss.write_index(index, index_file)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ Indexed {len(new_vectors)} new sentences. Updated FAISS index and metadata.")
    else:
        logging.info("✅ No new sentences to process. Index remains unchanged.")

def update_metadata_with_tfidf(metadata):
    sentences = [entry["sentence"] for entry in metadata]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_weights = tfidf_matrix.sum(axis=1).A1
    for entry, weight in zip(metadata, tfidf_weights):
        entry["tfidf_weight"] = weight
    return metadata

# -----------------------------
# RAG-Augmented Search Components
# -----------------------------
def build_contextual_chunks(metadata, window_size=2):
    """Create contextual chunks with surrounding sentences for enhanced context."""
    chunks = []
    for i, item in enumerate(metadata):
        start = max(0, i - window_size)
        end = min(len(metadata), i + window_size + 1)
        context = " ".join([metadata[j]["sentence"] for j in range(start, end)])
        chunks.append({
            "text": context,
            "source": item["file"],
            "sentence_hash": item["sentence_hash"]
        })
    return chunks

class RAGRetriever:
    def __init__(self, index_file, metadata_file, model,faiss_candidates =80):
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.chunks = build_contextual_chunks(self.metadata)
        self.model = model
        self.faiss_candidates = faiss_candidates
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._fit_tfidf()
    
    def _fit_tfidf(self):
        self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
    
    def hybrid_score(self, query_vec, text):
        """Calculate hybrid score combining semantic and lexical relevance."""
        # In interactive queries, disable progress bar.
        encoded = self.model.encode(text, show_progress_bar=False)
        semantic_score = float(np.dot(query_vec, encoded))
        lexical_score = self.vectorizer.transform([text]).sum(axis=1).A1[0]
        return 0.7 * semantic_score + 0.3 * lexical_score
    
    # In the RAGRetriever class:
    def retrieve(self, query, top_k=5):
        query_vec = self.model.encode(f"query: {query}", normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
        distances, indices = self.index.search(query_vec, self.faiss_candidates)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            hybrid = self.hybrid_score(query_vec, chunk["text"])
            results.append((hybrid, chunk))
        
        seen = set()
        final_results = []
        for hybrid, chunk in sorted(results, key=lambda x: x[0], reverse=True):
            if chunk["sentence_hash"] not in seen:
                seen.add(chunk["sentence_hash"])
                # Add the score to the chunk dictionary
                final_results.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "sentence_hash": chunk["sentence_hash"],
                    "score": hybrid  # This is the critical addition
                })
            if len(final_results) >= top_k:
                break
        return final_results

# -----------------------------
# Google LLM Generation Function
# -----------------------------
def google_text_generation(prompt, api_key, 
                           endpoint="https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent",
                           temperature=0.2, max_output_tokens=256):
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.95
        }
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    try:
        response = requests.post(endpoint, headers=headers, params=params, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logging.error(f"❌ Error during Google LLM text generation: {e}")
        return ""

def rag_generation(query, context_chunks, api_key):
    context_str = "\n\n".join([
        f"Excerpt {i+1} from {chunk['source']}:\n{chunk['text']}" 
        for i, chunk in enumerate(context_chunks)
    ])
    prompt = f"""**Job Requirement:** {query}

**Relevant Resume Excerpts:**
{context_str}

**Task:** 
1. Analyze if the candidate meets the requirement based ONLY on the excerpts.
2. Identify SPECIFIC matching qualifications with source citations (e.g., [Source 1]).
3. If no match exists, clearly state this.

**Response Guidelines:**
- Cite with [Source #] for each claim.
- Do not invent any qualifications.
- Highlight technical competencies and compare relevant experience."""
    
    return google_text_generation(prompt, api_key, temperature=0.1)

def rag_enhanced_search(query, model, index_file, metadata_file, top_k=5, api_key=""):
    retriever = RAGRetriever(
    index_file="faiss.index",
    metadata_file="faiss_metadata.json",
    model=model,
    faiss_candidates=80  # Set to 120/200 if you need even more
)
    context_chunks = retriever.retrieve(query, top_k=top_k)
    if not context_chunks:
        return "No relevant qualifications found in resumes."
    
    resume_contexts = defaultdict(list)
    for chunk in context_chunks:
        resume_contexts[chunk['source']].append(chunk)
    
    results = []
    for resume, chunks in resume_contexts.items():
        justification = rag_generation(query, chunks, api_key)
        results.append({
            "resume": resume,
            "match_score": len(chunks),
            "justification": justification,
            "source_chunks": chunks
        })
    return sorted(results, key=lambda x: x["match_score"], reverse=True)

# -----------------------------
# Hybrid Search Function with LLM caching
# -----------------------------
LLM_SCORE_CACHE = {}
CACHE_PATH = "llm_score_cache.json"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        LLM_SCORE_CACHE = json.load(f)

def make_cache_key(query, text):
    key_str = f"{query}::{text}"
    return hashlib.sha256(key_str.encode()).hexdigest()

def get_from_cache(query, text):
    key = make_cache_key(query, text)
    return LLM_SCORE_CACHE.get(key)

def store_to_cache(query, text, score, explanation):
    key = make_cache_key(query, text)
    LLM_SCORE_CACHE[key] = {"score": score, "explanation": explanation}
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(LLM_SCORE_CACHE, f, indent=2)

def normalize_scores(results, key="score"):
    scores = [r[key] for r in results]
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

def reciprocal_rank_fusion(rank, k=60):
    return 1.0 / (k + rank)

def score_relevance_with_llm(query, candidates, api_key):
    reranked = []
    for item in tqdm(candidates, desc="🧠 Reranking with LLM", leave=False):
        text = item["text"]
        cache_key = f"{query}|{item['sentence_hash']}"
        cached = get_from_cache(query, text)
        if cached:
            llm_score = cached["score"]
            explanation = cached["explanation"]
            logging.info("✅ Retrieved cached LLM score.")
        else:
            prompt = f"""Assess the relevance of the following resume excerpt for the query.
            
Query: "{query}"
            
Excerpt from {item['source']}:
"{text}"
            
Score the match from 0 (no match) to 1 (strong match). Respond only with a number."""
            score_text = google_text_generation(prompt, api_key, temperature=0.1, max_output_tokens=10)
            try:
                llm_score = float(score_text.strip())
                llm_score = max(0.0, min(llm_score, 1.0))
            except Exception:
                llm_score = 0.0
            explanation = score_text.strip()
            store_to_cache(query, text, llm_score, explanation)
        item["llm_score"] = llm_score
        item["llm_explanation"] = explanation
        reranked.append(item)
    return sorted(reranked, key=lambda x: -x["llm_score"])


def rag_generation_hybrid(query, context_chunks, api_key):
    """
    🔥 NEW FUNCTION: Specialized prompt for hybrid search results
    """
    context_str = "\n".join([
        f"- From {chunk['source']}: {chunk['text'][:200]}..." 
        for chunk in context_chunks
    ])
    
    prompt = f"""Evaluate this resume excerpt against the job requirement:

**Job Requirement:** {query}

**Resume Excerpt:**
{context_str}

**Tasks:**
1. Does this excerpt show relevant experience? Answer Yes/No
2. List SPECIFIC matching skills/experience
3. Note any missing requirements"""

    return google_text_generation(prompt, api_key)


def hybrid_search(query, vector_results, keyword_results, api_key=None, top_k=10, rerank_with_llm=True):
    """
    Combines vector and keyword search results using:
    - Reciprocal Rank Fusion for scoring
    - Deduplication based on sentence_hash
    - Optional re-ranking using LLM-based relevance assessment (with caching)
    """
    norm_vec_scores = normalize_scores(vector_results, key="score")
    norm_kwd_scores = normalize_scores(keyword_results, key="score")
    
    combined = defaultdict(lambda: {"score": 0.0, "source": "", "text": "", "sentence_hash": ""})
    for rank, (item, score) in enumerate(zip(vector_results, norm_vec_scores)):
        hash_id = item.get("sentence_hash") or hash(item["text"])
        combined[hash_id]["score"] += reciprocal_rank_fusion(rank)
        combined[hash_id].update(item)
    for rank, (item, score) in enumerate(zip(keyword_results, norm_kwd_scores)):
        hash_id = item.get("sentence_hash") or hash(item["text"])
        combined[hash_id]["score"] += reciprocal_rank_fusion(rank)
        combined[hash_id].update(item)
    
    fused_results = sorted(combined.values(), key=lambda x: -x["score"])
    top_candidates = fused_results[:top_k]
    
    if rerank_with_llm and api_key:
        reranked = score_relevance_with_llm(query, top_candidates, api_key)
        
        # New justification generation for each result
        justified_results = []
        for item in reranked:
            justification = rag_generation_hybrid(query, [item], api_key)  # 🔥 CHANGED FUNCTION
            justified_results.append({
                **item,
                "justification": justification
            })
        return justified_results
    return top_candidates

# -----------------------------
# Main Function: Bringing Everything Together
# -----------------------------
if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(
        description="RAG-Enhanced Resume Search Engine with Hybrid Search and LLM-based Re-ranking (with caching and progress bars during build)."
    )
    parser_cli.add_argument("--pdf_dir", type=str, default="resumes", help="Directory containing resume files")
    parser_cli.add_argument("--tika_url", type=str, default="http://localhost:9998/rmeta/text", help="Tika endpoint URL")
    parser_cli.add_argument("--index_file", type=str, default="faiss.index", help="FAISS index file")
    parser_cli.add_argument("--metadata_file", type=str, default="faiss_metadata.json", help="Metadata mapping file")
    parser_cli.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index from scratch")
    parser_cli.add_argument("--google_api_key", type=str, default="", help="Google API key for text generation")
    args = parser_cli.parse_args()
    
    if args.rebuild:
        if os.path.exists(args.index_file):
            os.remove(args.index_file)
        if os.path.exists(args.metadata_file):
            os.remove(args.metadata_file)
        logging.info("🔄 Rebuild selected: existing index and metadata removed.")
    
    # Initialize the SentenceTransformer model.
    model = SentenceTransformer("intfloat/e5-large")
    
    # Process resume files (with progress bar during build)
    logging.info("🚀 Starting resume file processing...")
    new_results = process_all_files(args.pdf_dir, model, args.tika_url, args.metadata_file)
    
    # Build or update the FAISS index (with progress bars)
    build_incremental_index(new_results, model, args.index_file, args.metadata_file)
    
    # Interactive loop for search queries.
    print("\nEnter a job requirement or position to search resumes.")
    print("Prefix the query with 'hybrid:' to trigger hybrid search. (Type 'exit' to quit.)")
    
    while True:
        query_input = input("\n🔎 Enter a query (or 'exit'): ").strip()
        if query_input.lower() == "exit":
            break
        
        # Determine if we use hybrid or RAG mode based on prefix.
        if query_input.lower().startswith("hybrid:"):
            actual_query = query_input[7:].strip()
            retriever = RAGRetriever(args.index_file, args.metadata_file, model)
            vector_results = retriever.retrieve(actual_query, top_k=10)
            # For keyword results, you can implement your own; here we simply reuse vector results.
            keyword_results = vector_results  
            if not args.google_api_key:
                print("❌ Google API key not provided. Use the --google_api_key argument.")
                continue
            hybrid_results = hybrid_search(actual_query, vector_results, keyword_results, api_key=args.google_api_key, top_k=5, rerank_with_llm=True)
            print(f"\n🤖 Hybrid Search Results for: {actual_query}")
            for idx, res in enumerate(hybrid_results, 1):
                print(f"\n🏆 Result {idx} from {res['source']}:")
                print(f"Text: {res['text'][:150]}...")
                print(f"🔢 Score: {res['score']:.4f}")
                print(f"💡 Justification: {res.get('justification', 'No justification available')}")  # 🔥 NEW OUTPUT
                print("-" * 80)
        else:
            # Default to RAG-enhanced search.
            actual_query = query_input
            if not args.google_api_key:
                print("❌ Google API key not provided. Use the --google_api_key argument.")
                continue
            results = rag_enhanced_search(actual_query, model, args.index_file, args.metadata_file, top_k=5, api_key=args.google_api_key)
            print(f"\n🤖 RAG-Enhanced Search Results for: {actual_query}")
            for idx, result in enumerate(results, 1):
                print(f"\n🏆 Match #{idx}: {result['resume']}")
                print(f"Match Strength (excerpts count): {result['match_score']}")
                print("Justification:")
                print(result['justification'])
                print("Supporting Context:")
                for chunk in result["source_chunks"]:
                    print(f"   - From {chunk['source']}: {chunk['text'][:150]}...")
