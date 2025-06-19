 # backend/app/main.py

from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import json
import os
import sys
import google.generativeai as genai

# This allows importing from your existing llm_service.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# We will use the functions you already have
from llm_service import generate_json_response, clean_gemini_response

# -- Initialize the FastAPI App --
app = FastAPI(
    title="Intelligent Universal Prompt Table Generator API",
    version="1.5" # Version bump for robust AI Chaining!
)

# -- Configure CORS --
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
#        HELPER FUNCTIONS FOR THE NEW "AI CHAIN"
# ==========================================================

def get_pdf_columns_with_llm(raw_text: str) -> list[str]:
    """
    AI CHAIN STEP 1: Makes a small, fast AI call to identify column headers from the PDF.
    """
    if not raw_text.strip(): return []
    try:
        # A very focused prompt to extract only the column headers
        parsing_prompt = f"""Analyze the start of this text from a PDF and identify the column headers. Return a single, flat JSON array of strings with the header names. Example: ["SI. No.", "USN", "Name"]. Ignore document titles. Text: --- {raw_text[:1000]} ---"""
        model = genai.GenerativeModel("gemini-1.5-flash-latest") # Use a fast model for this simple task
        response = model.generate_content(
            parsing_prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        headers = json.loads(response.text)
        return headers if isinstance(headers, list) else []
    except Exception as e:
        print(f"Error getting columns from PDF: {e}")
        return []

def populate_data_with_llm(raw_text: str, schema: dict) -> list:
    """
    AI CHAIN STEP 3: Uses a focused AI call to parse the full PDF text against a final, correct schema.
    """
    if not raw_text.strip(): return [{}]
    try:
        # A focused prompt to parse data against a pre-defined schema
        parsing_prompt = f"""
        You are an expert data parser. Parse the 'Raw Text' and structure it into a JSON array that fits the provided 'JSON Schema'. Map the data for each row to the correct column `id`. Ignore headers in the raw text.

        **JSON Schema to follow:**
        ```json
        {json.dumps(schema, indent=2)}
        ```

        **Raw Text to parse:**
        ---
        {raw_text}
        ---

        Your output must be ONLY the JSON array of objects.
        """
        # We reuse your main llm_service function for this complex parsing task
        # We wrap the prompt in the format our llm_service expects
        result = generate_json_response(f"USER PROMPT: {parsing_prompt}")
        # The AI is instructed to return schema and data, but we only need the data here
        return result.get("tableData", [{}])
    except Exception as e:
        print(f"Error populating data with LLM: {e}")
        return [{}]

# ==========================================================
#        MAIN ENDPOINT ORCHESTRATING THE AI CHAIN
# ==========================================================

@app.post("/generate-table")
async def generate_table_endpoint(
    prompt: str = Form(...),
    file: UploadFile = File(None)
):
    final_prompt_for_schema = prompt
    raw_pdf_text = ""
    table_data = [{}]

    # --- AI CHAIN STEP 1: Analyze PDF and Create "Super-Prompt" ---
    if file and file.filename:
        try:
            pdf_stream = file.file.read()
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            raw_pdf_text = "".join(page.get_text() for page in doc)
            doc.close()
            
            pdf_columns = get_pdf_columns_with_llm(raw_pdf_text)
            if pdf_columns:
                print(f"Detected columns from PDF: {pdf_columns}")
                columns_text = ", ".join(f'"{c}"' for c in pdf_columns)
                super_prompt_addition = f" The table must also include these columns at the beginning: {columns_text}. Make the most appropriate column (like USN or ID) the primary key."
                final_prompt_for_schema += super_prompt_addition
        except Exception as e:
            print(f"Failed to read or analyze PDF: {e}")

    # --- AI CHAIN STEP 2: Generate the Final, Complete Schema ---
    print(f"Generating schema with combined prompt...")
    # We ask the AI to generate the full response, but we primarily need the schema from this step.
    schema_generation_prompt = f"USER PROMPT: {final_prompt_for_schema}\n\n--- PDF TEXT ---\n{raw_pdf_text}"
    schema_result = generate_json_response(schema_generation_prompt)
    
    if "error" in schema_result or "schema" not in schema_result:
        raise HTTPException(status_code=500, detail="Failed to generate table schema.")
    
    schema = schema_result["schema"]

    # --- AI CHAIN STEP 3: Populate Data Using the Final Schema ---
    if raw_pdf_text:
        print("Populating data against the final schema...")
        # Now we call our second, specialized data-population function
        table_data = populate_data_with_llm(raw_pdf_text, schema)
    
    # --- Final Step: Return the complete, correct result ---
    return { "schema": schema, "tableData": table_data }

# Other endpoints remain the same
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if os.path.exists("static/favicon.ico"): return FileResponse("static/favicon.ico")
    return {"status": "no favicon"}

@app.get("/")
def root(): return {"status": "ok"}