from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()


def generate_answer(question, results):
    """
    Generate an answer using the retrieved documents.
    """

    if not results:
        return "I couldn't find any relevant information in the documents."

    context_parts = []

    for result in results:
        source = result.metadata.get("source", "Unknown")
        page = result.metadata.get("page", "Unknown")
        content = result.page_content

        context_parts.append(
            f"Source: {source}, Page: {page}\n{content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information provided
in the document context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided documents."

Do not make up information.

User question:
{question}

Document context:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text