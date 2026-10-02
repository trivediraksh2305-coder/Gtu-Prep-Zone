import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()  # loads .env when running locally; on Render, env vars come from dashboard

app = Flask(__name__)

# --------------------------------------------------------------------------
# CORS - only allow requests from your actual site(s)
# --------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://trivediraksh2305-coder.github.io",
    "https://www.gtuprepzone.com",
    "https://gtuprepzone.com",
    "http://127.0.0.1:5500",   # local testing with VSCode Live Server, remove if unused
    "http://localhost:5500",
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_PDF_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_CHARACTERS = 60000

SYSTEM_PROMPT = """You are GTU Prep Zone AI, an AI study assistant for Gujarat Technological University (GTU) students.

The student may or may not have uploaded a GTU question paper. If a question paper is provided in the user message, use it as the main context. If no question paper is provided, answer the student's question using your own knowledge of GTU subjects and exam patterns.

IMPORTANT RULES:

1. Answer the question asked by the student. Nothing else.
2. Follow GTU examination style.
3. Use simple and clear student-friendly language.
4. Structure answers using markdown: ## for section headings, **bold** for key terms, and proper markdown tables (with a header row and a |---|---| separator row) only when comparing 2+ items.
5. If the question requires a definition, give a clear one-line definition first, then explain.
6. If the question requires an explanation, give enough detail for the marks, organized as short paragraphs or bullet points — not giant dense blocks of text.
7. If a diagram is useful, describe it briefly in words rather than attempting ASCII art.
8. For numerical problems, show each step on its own line.
9. Do not invent information that isn't in the uploaded question paper.
10. If the requested question cannot be found or understood from the uploaded paper, say so briefly and stop.
11. Do not refer to yourself as ChatGPT.

STRICT FORMATTING RULES - FOLLOW EXACTLY:
- Do NOT add a "---" horizontal rule anywhere in the answer.
- Do NOT end the answer with a summary of how you structured it, a recap outline, a numbered meta-list like "Answer Structure for X Marks", or any note about exam suitability. Just give the answer and stop.
- Do NOT restate the question before answering.
- Every markdown table must have exactly one header row, one separator row, and complete data rows — never leave a table row incomplete.
- Never output an empty or incomplete numbered/bulleted list item (e.g. a list item with no text after it).

MARKS GUIDELINE:

2-3 marks:
Give a short and precise answer.

4-5 marks:
Give a moderate explanation with important points.

6-7 marks:
Give a detailed examination-ready answer with suitable examples/diagrams where appropriate.
"""


def error_response(message, status=200):
    # Kept status=200 by default to match old PHP behavior where frontend
    # checks data.success rather than HTTP status. Change to 400 if you
    # update the frontend's error handling later.
    return jsonify({"success": False, "message": message}), status


@app.route("/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return error_response("Server misconfiguration: missing API key.")

    user_question = request.form.get("message", "").strip()
    if not user_question:
        return error_response("Please enter a question.")

    pdf_text = ""

    uploaded_file = request.files.get("pdf")

    if uploaded_file and uploaded_file.filename:
        filename = uploaded_file.filename.lower()
        if not filename.endswith(".pdf"):
            return error_response("Only PDF files are allowed.")

        uploaded_file.seek(0, os.SEEK_END)
        size = uploaded_file.tell()
        uploaded_file.seek(0)
        if size > MAX_PDF_SIZE:
            return error_response("PDF is too large. Maximum allowed size is 15 MB.")

        try:
            reader = PdfReader(uploaded_file)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            pdf_text = "\n".join(pages_text).strip()
        except Exception as e:
            return error_response(f"Could not read the PDF. {str(e)}")

        if not pdf_text:
            return error_response(
                "No readable text was found in this PDF. The PDF may be scanned/image-based."
            )
    else:
        # No new PDF this request - use paper text from an earlier upload if
        # the frontend sent one back. It's fine if this is empty; the student
        # may just be asking a general GTU question with no paper attached.
        pdf_text = request.form.get("paper_text", "").strip()

    if len(pdf_text) > MAX_CHARACTERS:
        pdf_text = pdf_text[:MAX_CHARACTERS] + "\n\n[PDF text truncated because it was very large.]"

    if pdf_text:
        user_prompt = f"""Here is the text extracted from the student's uploaded GTU question paper:

---------------- QUESTION PAPER START ----------------

{pdf_text}

---------------- QUESTION PAPER END ----------------

Student's question:

{user_question}

Answer the student's question according to the GTU examination pattern.
"""
    else:
        user_prompt = f"""The student has not uploaded a question paper for this question. Answer using your own GTU-relevant knowledge instead.

Student's question:

{user_question}

Answer according to the GTU examination pattern.
"""

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        return error_response(f"Could not connect to Groq API. {str(e)}")

    if not resp.ok:
        try:
            err = resp.json().get("error", {}).get("message", "Groq API returned an error.")
        except ValueError:
            err = "Groq API returned an error."
        return error_response(err)

    data = resp.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    if not answer:
        return error_response("Groq returned an empty answer.")

    return jsonify({
        "success": True,
        "message": "Answer generated successfully.",
        "answer": answer,
        "paper_text": pdf_text,  # frontend stores this and resends on follow-up questions
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)