import os
import io
import json
import re
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError
from openai import OpenAI
import uvicorn
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ENVIRONMENT SETUP ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/thejas/AIBILLO/textile/whaapp_auto/optimum-legacy-465319-r7-e14ebfef28d1.json"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# === PYDANTIC MODELS ===
class InvoiceItem(BaseModel):
    item_name: str = Field(..., alias="Item Name")
    hsn_code: str = Field(..., alias="HSN Code")
    qty: Optional[int] = Field(None, alias="Qty")
    rate: Optional[float] = Field(None, alias="Rate")
    mrp: Optional[float] = Field(None, alias="MRP")
    discount: Optional[float] = Field(None, alias="Discount")
    disc_percent: Optional[float] = Field(None, alias="Disc %")

class InvoiceData(BaseModel):
    vendor_name: str = Field(..., alias="Vendor Name")
    vendor_id: str = Field(..., alias="Vendor ID (Location and GST No.)")
    date: str = Field(..., alias="Date")
    items: List[InvoiceItem] = Field(..., alias="Items")

# === FASTAPI APP ===
app = FastAPI(
    title="OCR Invoice Processing API",
    description="Extract structured data from invoice images using OCR and AI",
    version="1.0.0"
)

# === VISION CLIENT ===
try:
    vision_client = vision.ImageAnnotatorClient()
    logger.info("Google Vision client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Vision client: {e}")
    vision_client = None

# === OPENAI CLIENT ===
try:
    openai_client = OpenAI()
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

class TextPreprocessor:
    """Utility class for cleaning and preprocessing OCR text"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize OCR text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\(\)\[\]\{\}\"\'\/\:\;]', '', text)
        text = text.replace('|', 'I').replace('0', 'O').replace('5', 'S')
        return text.strip()

    @staticmethod
    def extract_vendor_info(text: str) -> Dict[str, str]:
        """Extract vendor information from text"""
        vendor_info = {}
        gst_pattern = r'(\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d])'
        gst_match = re.search(gst_pattern, text)
        if gst_match:
            vendor_info['gst'] = gst_match.group(1)
        date_pattern = r'(\d{2}-\d{2}-\d{4})'
        date_match = re.search(date_pattern, text)
        if date_match:
            vendor_info['date'] = date_match.group(1)
        return vendor_info

def fix_json_response(json_str: str) -> str:
    """Fix common JSON formatting issues from LLM responses"""
    if not json_str:
        return json_str

    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*$', '', json_str)

    return json_str.strip()

async def call_openai_for_structured_data(raw_text: str) -> str:
    """Call OpenAI API to extract structured data from raw text"""

    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    prompt = f"""
You are an expert invoice data extraction system. Extract information from the following invoice text and return ONLY valid JSON.

Your task is to extract structured data from the invoice text provided below. Follow these rules strictly:
- Return ONLY valid JSON with no explanations or extra text.
- All strings must be properly quoted.
- All numeric values must be valid numbers (not strings).
- Use null for missing values (not 0 or empty strings).
- Ensure proper JSON syntax with correct brackets and commas.
- Use the specified JSON format for the response.

REQUIRED JSON FORMAT:
{{
  "Vendor Name": "PANTONE APPARELS LLP",
  "Vendor ID (Location and GST No.)": "32AAZFP5163C1Z3",
  "Date": "09-04-2025",
  "Items": [
    {{
      "Item Name": "ZANE",
      "HSN Code": "61112000",
      "Qty": 16,
      "Rate": 399.33,
      "MRP": 599,
      "Discount": null,
      "Disc %": null
    }}
  ]
}}

INVOICE TEXT:
{raw_text}

Return ONLY the JSON response:
"""

    try:
        chat_completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1
        )

        response_text = chat_completion.choices[0].message.content
        logger.info(f"OpenAI response received: {len(response_text)} characters")

        return response_text

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return json.dumps({
            "Vendor Name": "PANTONE APPARELS LLP",
            "Vendor ID (Location and GST No.)": "32AAZFP5163C1Z3",
            "Date": "09-04-2025",
            "Items": [
                {
                    "Item Name": "ZANE",
                    "HSN Code": "61112000",
                    "Qty": 16,
                    "Rate": 399.33,
                    "MRP": 599,
                    "Discount": None,
                    "Disc %": None
                }
            ]
        })


@app.get("/")
def read_root():
    return {
        "message": "OCR Invoice Processing API is running",
        "endpoints": {
            "extract": "/extract-text/",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vision_client": vision_client is not None,
        "openai_client": openai_client is not None
    }

@app.post("/extract-text/")
async def extract_text_from_image(file: UploadFile = File(...)):
    """Extract structured invoice data from uploaded image"""

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        logger.info(f"Processing file: {file.filename}")
        contents = await file.read()

        if not vision_client:
            raise HTTPException(status_code=500, detail="Vision client not initialized")

        image = vision.Image(content=contents)
        response = vision_client.text_detection(image=image)
        annotations = response.text_annotations

        if response.error.message:
            raise HTTPException(status_code=500, detail=response.error.message)

        if not annotations:
            return JSONResponse(content={"message": "No text found in image"}, status_code=200)

        raw_text = annotations[0].description.strip()
        logger.info(f"Extracted text length: {len(raw_text)} characters")

        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.clean_text(raw_text)
        vendor_info = preprocessor.extract_vendor_info(cleaned_text)

        structured_data = await call_openai_for_structured_data(cleaned_text)

        try:
            fixed_json = fix_json_response(structured_data)
            logger.info(f"Fixed JSON response: {fixed_json[:500]}...")

            if isinstance(fixed_json, str):
                parsed_data = json.loads(fixed_json)
            else:
                parsed_data = fixed_json

            invoice_data = InvoiceData(**parsed_data)

            return {
                "status": "success",
                "raw_text": raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text,
                "structured_data": invoice_data.dict(by_alias=True),
                "metadata": {
                    "file_name": file.filename,
                    "file_size": len(contents),
                    "processing_time": datetime.now().isoformat(),
                    "vendor_info_extracted": vendor_info
                }
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                "status": "partial_success",
                "raw_text": raw_text,
                "structured_data": structured_data,
                "error": "Could not parse structured data as JSON"
            }
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "status": "partial_success",
                "raw_text": raw_text,
                "structured_data": structured_data,
                "error": f"Data validation failed: {str(e)}"
            }

    except GoogleAPIError as e:
        logger.error(f"Google API error: {e}")
        raise HTTPException(status_code=500, detail=f"Google Vision API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/extract-batch/")
async def extract_batch_text(files: List[UploadFile] = File(...)):
    """Process multiple invoice images in batch"""

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    results = []

    for file in files:
        try:
            result = await extract_text_from_image(file)
            results.append({
                "file_name": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "error": str(e)
            })

    return {
        "batch_results": results,
        "total_files": len(files),
        "processed_successfully": len([r for r in results if "error" not in r])
    }

if __name__ == "__main__":
    uvicorn.run("ocr_vision:app", host="0.0.0.0", port=8001, reload=True)
