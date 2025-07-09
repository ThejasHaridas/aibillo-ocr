import os
import io
import json
import re
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError
from groq import Groq
from pdf2image import convert_from_bytes
import uvicorn
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ENV VARS ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "optimum-legacy-465319-r7-e14ebfef28d1.json"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your_fallback_groq_key")
GROQ_MODEL = "llama-3.1-8b-instant"

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

# === APP ===
app = FastAPI(
    title="OCR Invoice Processing API",
    description="Extract structured data from invoice images & PDFs using OCR + LLM",
    version="1.0.0"
)

# === CLIENTS ===
try:
    vision_client = vision.ImageAnnotatorClient()
    logger.info("Google Vision client initialized")
except Exception as e:
    logger.error(f"Vision client error: {e}")
    vision_client = None

try:
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    logger.info("Groq client initialized")
except Exception as e:
    logger.error(f"Groq client error: {e}")
    groq_client = None

# === UTILS ===
class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\(\)\[\]\{\}\"\'\/\:\;]', '', text)
        text = text.replace('|', 'I').replace('0', 'O').replace('5', 'S')
        return text.strip()

    @staticmethod
    def extract_vendor_info(text: str) -> Dict[str, str]:
        vendor_info = {}
        gst_pattern = r'(\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d])'
        date_pattern = r'(\d{2}-\d{2}-\d{4})'
        gst_match = re.search(gst_pattern, text)
        date_match = re.search(date_pattern, text)
        if gst_match:
            vendor_info['gst'] = gst_match.group(1)
        if date_match:
            vendor_info['date'] = date_match.group(1)
        return vendor_info

def fix_json_response(json_str: str) -> str:
    if not json_str:
        return json_str
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*$', '', json_str)
    json_str = json_str.replace('61112OOO', '61112000')
    json_str = re.sub(r'({\s*{\s*"Item Name")', r'{"Item Name"', json_str)
    json_str = json_str.replace('"Item":Name"', '"Item Name"')
    json_str = re.sub(r'",\s*}', '"}', json_str)
    return json_str

async def call_groq(raw_text: str) -> str:
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized")

    prompt = f"""
You are an expert invoice extractor.
Extract ONLY valid JSON in this format:
{{
  "Vendor Name": "...",
  "Vendor ID (Location and GST No.)": "...",
  "Date": "DD-MM-YYYY",
  "Items": [
    {{
      "Item Name": "...",
      "HSN Code": "61112000",
      "Qty": 10,
      "Rate": 399.33,
      "MRP": 599,
      "Discount": null,
      "Disc %": null
    }}
  ]
}}
Text:
{raw_text}
Return ONLY JSON.
"""
    chat_completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000
    )
    return chat_completion.choices[0].message.content

async def process_text_and_structure(raw_text: str, meta: dict):
    pre = TextPreprocessor()
    cleaned = pre.clean_text(raw_text)
    vendor_info = pre.extract_vendor_info(cleaned)
    structured = await call_groq(cleaned)
    try:
        fixed = fix_json_response(structured)
        parsed = json.loads(fixed)
        invoice = InvoiceData(**parsed)
        return {
            "status": "success",
            "raw_text_sample": cleaned[:1000] + "..." if len(cleaned) > 1000 else cleaned,
            "structured_data": invoice.dict(by_alias=True),
            "vendor_info_extracted": vendor_info,
            "metadata": meta
        }
    except Exception as e:
        logger.error(f"Validation/JSON error: {e}")
        return {
            "status": "partial_success",
            "raw_text_sample": cleaned,
            "structured_data": structured,
            "error": str(e),
            "vendor_info_extracted": vendor_info,
            "metadata": meta
        }

# === ROUTES ===
@app.get("/")
def root():
    return {"message": "API live", "endpoints": ["/extract-text", "/extract-pdf", "/extract-batch"]}

@app.get("/health")
def health():
    return {"vision_client": vision_client is not None, "groq_client": groq_client is not None}

@app.post("/extract-text/")
async def extract_text_from_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    image = vision.Image(content=contents)
    response = vision_client.text_detection(image=image)
    if response.error.message:
        raise HTTPException(status_code=500, detail=response.error.message)
    annotations = response.text_annotations
    if not annotations:
        return JSONResponse(content={"message": "No text found"}, status_code=200)
    raw_text = annotations[0].description
    meta = {"file_name": file.filename, "file_size": len(contents), "processing_time": datetime.now().isoformat()}
    return await process_text_and_structure(raw_text, meta)

@app.post("/extract-pdf/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    contents = await file.read()
    images = convert_from_bytes(contents)
    logger.info(f"PDF -> {len(images)} pages")
    full_text = ""
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        image = vision.Image(content=buf.getvalue())
        response = vision_client.document_text_detection(image=image)
        if response.error.message:
            raise HTTPException(status_code=500, detail=response.error.message)
        annotations = response.text_annotations
        if annotations:
            full_text += annotations[0].description + "\n"
    if not full_text.strip():
        return JSONResponse(content={"message": "No text found in PDF"}, status_code=200)
    meta = {"file_name": file.filename, "pages": len(images), "file_size": len(contents), "processing_time": datetime.now().isoformat()}
    return await process_text_and_structure(full_text, meta)

@app.post("/extract-batch/")
async def extract_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            if file.filename.lower().endswith('.pdf'):
                res = await extract_text_from_pdf(file)
            else:
                res = await extract_text_from_image(file)
            results.append({"file": file.filename, "result": res})
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
    return {"batch_results": results, "count": len(files)}

if __name__ == "__main__":
    uvicorn.run("ocr_vision:app", host="0.0.0.0", port=8001, reload=True)
