import json
import os
import numpy as np
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data.json')
 

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
EMBEDDING_URL = "https://models.inference.ai.azure.com/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
 
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)
 
questions = [item['question'] for item in qa_data]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)
 
def get_query_embedding(text):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    body = {
        "model": EMBEDDING_MODEL,
        "input": text[:2000]
    }
    response = requests.post(EMBEDDING_URL, headers=headers, json=body, timeout=10)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
 
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0
 
def home_view(request):
    return render(request, 'search/index.html')
 
@csrf_exempt
def api_search(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            query = body.get('question', '').strip()
        except Exception:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        query = request.GET.get('q', '').strip()
 
    if not query:
        return JsonResponse({'results': []})
 
    use_github = False
    top_indices = []
 
    try:
        query_embedding = get_query_embedding(query)
        scored = []
        for i, item in enumerate(qa_data):
            if item.get('embedding'):
                score = cosine_sim(query_embedding, item['embedding'])
                scored.append((score, i))
 
        if scored:
            scored.sort(reverse=True)
            top_indices = [i for _, i in scored[:5]]
            similarities = [s for s, _ in scored[:5]]
            use_github = True
 
    except Exception:
        pass
 
    if not use_github:
        query_vec = vectorizer.transform([query])
        sims = sklearn_cosine(query_vec, question_vectors).flatten()
        top_indices = sims.argsort()[::-1][:5]
        similarities = [float(sims[i]) for i in top_indices]
 
    results = []
    for idx, sim in zip(top_indices, similarities):
        results.append({
            'question': qa_data[idx]['question'],
            'answer': qa_data[idx]['answer'],
            'similarity': round(float(sim), 3),
            'engine': 'GitHub Models' if use_github else 'TF-IDF'
        })
 
    return JsonResponse({
        'results': results,
        'engine': 'GitHub Models (semantic)' if use_github else 'TF-IDF (keyword)'
    })