import json
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data.json')

print("Loading model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded!")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

print(f"Loaded {len(qa_data)} Q&A pairs")

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

    query_embedding = model.encode(query).tolist()

    scored = []
    for i, item in enumerate(qa_data):
        if item.get('embedding'):
            score = cosine_sim(query_embedding, item['embedding'])
            scored.append((score, i))

    scored.sort(reverse=True)
    top5 = scored[:5]

    results = []
    for score, idx in top5:
        results.append({
            'question': qa_data[idx]['question'],
            'answer': qa_data[idx]['answer'],
            'similarity': round(float(score), 3),
            'engine': 'Semantic Search'
        })

    return JsonResponse({
        'results': results,
        'engine': 'Semantic Search (all-MiniLM-L6-v2)'
    })