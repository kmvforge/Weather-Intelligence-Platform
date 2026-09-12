import requests
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from retriever import retrieve_context


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def generate_answer(question, additional_context=""):
    """Generate a grounded answer using project knowledge and optional prediction context."""

    context = retrieve_context(question)

    if additional_context:
        context += "\n\n[CURRENT MODEL PREDICTION]\n" + additional_context

    prompt = f"""
You are the Weather Intelligence Assistant for an AI weather
intelligence project.

Answer the user's question using the PROJECT CONTEXT provided below.

Rules:
1. Use only information supported by the project context.
2. Do not invent statistics, measurements, model results, or facts.
3. Keep the answer concise and clear.
4. Use bullet points when useful.
5. If numerical results are available, include the relevant values.
6. Clearly distinguish historical observations from ML predictions.
7. Never invent a prediction value.
8. If a CURRENT MODEL PREDICTION is provided, use that exact value.
9. If the context does not contain enough information to answer the
   question, say:
   "I don't have enough information in the project knowledge base
   to answer that."

PROJECT CONTEXT
---------------
{context}

USER QUESTION
-------------
{question}

ANSWER
------
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()