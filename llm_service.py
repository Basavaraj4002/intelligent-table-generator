 # backend/app/llm_service.py

import os
import json
import time
import random
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded

# ----------------------------- ENV SETUP -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("ERROR: GOOGLE_API_KEY is not set in the .env file.")

# ----------------------------- LOGGER -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------- GEMINI CONFIG -----------------------------
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise RuntimeError(f"ERROR: Failed to configure Gemini API: {e}")

# ----------------------------- SYSTEM PROMPT -----------------------------
SYSTEM_INSTRUCTIONS = """
You are an expert data structuring assistant. Your task is to convert a user's natural language prompt and any provided text data into a single, structured JSON object.

**Your output MUST be a single JSON object with two main keys: "schema" and "tableData".**

**1. The "schema" Object:**
This defines the table structure. It should contain a "tableName" and a "columns" array. Each object in the "columns" array must have:
- `id`: A unique snake_case identifier.
- `header`: The human-readable column name.
- `type`: 'text', 'number', 'date', or 'boolean'.
- `isPrimaryKey`: `true` for one column only (usually the first unique identifier like a USN or ID).
- `isEditable`: `true` for data entry columns, `false` for calculated columns.
- `formula`: (Optional) A string like "SUM(col_id1, col_id2)" for calculated columns.
- `maxValue`: (Optional) A number if the prompt specifies a limit (e.g., "out of 10"). For number columns with a limit, also add "(Max: 10)" to the `header`.

**2. The "tableData" Array:**
This is an array of objects, where each object represents a row of data.

**CRITICAL RULES:**
- **IF a section labeled '--- PDF TEXT ---' is provided in the user's prompt:**
  - You MUST parse this text to populate the `tableData` array.
  - Each object in the array should map the data to the correct `id` from the schema you just defined.
  - The schema itself MUST include columns for all the data found in the PDF (e.g., "SI. No.", "USN", "Name").
- **IF NO '--- PDF TEXT ---' section is provided:**
  - You MUST return an empty array for `tableData`: `[]`.

**Final Output:** Your entire response must be ONLY the raw JSON object. Do not add any other text or markdown fences.
"""

# ==========================================================
#        THIS IS THE HELPER FUNCTION THAT WAS MISSING
# ==========================================================
def clean_gemini_response(text: str) -> str:
    """
    Removes markdown fences (```json ... ```) that the AI sometimes adds
    to its JSON response, making it safe to parse.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

# ----------------------------- MAIN FUNCTION -----------------------------
def generate_json_response(
    full_prompt: str,
    max_retries: int = 3,
    initial_backoff: float = 5.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    model_name: str = "models/gemini-1.5-flash-latest"
) -> dict:
    """
    Takes a combined prompt and gets a single JSON object containing
    both schema and tableData, with retry logic.
    """
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return {"error": "Failed to initialize LLM model.", "details": str(e)}

    retries_count = 0
    current_backoff = initial_backoff
    last_exception = None

    while retries_count <= max_retries:
        try:
            logger.info(f"Attempt {retries_count + 1}/{max_retries + 1} for combined prompt...")
            response = model.generate_content(
                [SYSTEM_INSTRUCTIONS, full_prompt],
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                ),
                request_options={"timeout": 60}
            )
            parsed_json = json.loads(response.text)
            return parsed_json
        except (ResourceExhausted, DeadlineExceeded) as e:
            last_exception = e
            if retries_count == max_retries:
                logger.error(f"Max retries reached ({type(e).__name__}): {e}")
                break
            wait_time = min(current_backoff + random.uniform(0, 1.0), max_backoff)
            logger.warning(f"{type(e).__name__} hit. Retrying in {wait_time:.2f}s.")
            time.sleep(wait_time)
            current_backoff *= backoff_factor
            retries_count += 1
        except json.JSONDecodeError as decode_error:
            logger.error(f"Invalid JSON response from LLM: {decode_error}")
            json_response_string = locals().get('response', None)
            if json_response_string:
                logger.debug(f"Raw response: {json_response_string.text}")
            return {"error": "Failed to parse Gemini response", "details": str(decode_error)}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": "Unexpected error during LLM generation", "details": str(e)}

    return {
        "error": "LLM request failed after multiple retries.",
        "details": str(last_exception) if last_exception else "Unknown error."
    }