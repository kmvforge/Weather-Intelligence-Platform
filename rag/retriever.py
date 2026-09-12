from pathlib import Path
import re


KNOWLEDGE_FILE = Path(__file__).parent / "weather_knowledge.txt"


def load_knowledge():
    """Load the weather knowledge base."""
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
        return file.read()


def split_sections(text):
    """Split the knowledge base into [SECTION] blocks."""
    pattern = r"\[(.*?)\]\n(.*?)(?=\n\[|\Z)"

    matches = re.findall(pattern, text, re.DOTALL)

    sections = []

    for title, content in matches:
        sections.append({
            "title": title.strip(),
            "content": content.strip()
        })

    return sections


def tokenize(text):
    """Convert text into searchable words."""
    return set(
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", text)
        if len(word) > 2
    )


def retrieve_context(query, max_sections=2):
    """
    Retrieve the most relevant complete sections
    from the weather knowledge base.
    """

    knowledge = load_knowledge()
    sections = split_sections(knowledge)

    query_lower = query.lower()
    query_words = tokenize(query)

    scored_sections = []

    for section in sections:

        title = section["title"]
        content = section["content"]

        searchable_text = (
            title + " " + content
        ).lower()

        section_words = tokenize(searchable_text)

        score = len(
            query_words.intersection(section_words)
        )

        # Strong boost for relevant section titles
        title_lower = title.lower()

        if "model" in query_lower:
            if "model performance" in title_lower:
                score += 10
            elif "model training" in title_lower:
                score += 3

        if "xgboost" in query_lower:
            if "model performance" in title_lower:
                score += 10
            elif "feature importance" in title_lower:
                score += 5

        if "feature" in query_lower:
            if "feature importance" in title_lower:
                score += 10

        if "temperature" in query_lower:
            if "exploratory data analysis" in title_lower:
                score += 8

        if "average" in query_lower:
            if "exploratory data analysis" in title_lower:
                score += 5

        if "rain" in query_lower or "rainfall" in query_lower:
            if "exploratory data analysis" in title_lower:
                score += 8

        if "api" in query_lower:
            if "fastapi" in title_lower:
                score += 10

        if "genai" in query_lower:
            if "genai" in title_lower:
                score += 10

        if "rag" in query_lower:
            if "rag" in title_lower:
                score += 10

        if score > 0:
            scored_sections.append(
                (score, section)
            )

    scored_sections.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    selected_sections = [
        section["title"] + "\n" + section["content"]
        for score, section in scored_sections[:max_sections]
    ]

    return "\n\n".join(selected_sections)


if __name__ == "__main__":

    questions = [
        "Which model performed best?",
        "What is the XGBoost MAE?",
        "What is the average temperature in Kochi?",
        "What are the most important features?",
        "What does the API do?",
        "What GenAI model is being used?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        context = retrieve_context(question)

        print("\nRetrieved Context:\n")
        print(context)