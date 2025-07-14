import os
import io
import json
import re
import asyncio
from typing import Dict, List, Optional
import easyocr
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field, ValidationError
import logging
import gc

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_IMAGE_SIZE = (512, 512)
MAX_TEXT_LENGTH = 500

# === PYDANTIC MODELS ===
class InvoiceItem(BaseModel):
    item_name: str = Field(..., alias="Item Name")
    hsn_code: Optional[str] = Field(None, alias="HSN Code")
    qty: Optional[int] = Field(None, alias="Qty")
    rate: Optional[float] = Field(None, alias="Rate")
    mrp: Optional[float] = Field(None, alias="MRP")

class InvoiceData(BaseModel):
    vendor_name: str = Field(..., alias="Vendor Name")
    GST_No: str = Field(..., alias="GST No")
    date: str = Field(..., alias="Date")
    items: List[InvoiceItem] = Field(..., alias="Items")

# === CLIENT INITIALIZATION ===
def get_easyocr_reader():
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        logger.critical("EasyOCR reader initialized on CPU with English only.")
        return reader
    except Exception as e:
        logger.critical(f"Failed to initialize EasyOCR: {e}")
        return None

def get_phi3_model():
    try:
        device = "cpu"
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME, trust_remote_code=True)
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                PHI3_MODEL_NAME,
                quantization_config=quantization_config,
                device_map={"": "cpu"},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                max_memory={0: "4GB"}
            )
        model.eval()
        logger.critical("Phi3 Mini model initialized with 4-bit quantization on CPU.")
        return model, tokenizer, device
    except Exception as e:
        logger.critical(f"Failed to initialize Phi3 Mini: {e}")
        return None, None, None

# Initialize clients
easyocr_reader = get_easyocr_reader()
phi3_model, phi3_tokenizer, device = get_phi3_model()

if not easyocr_reader or not phi3_model:
    logger.critical("Failed to initialize one or more components: EasyOCR or Phi3 model")
    raise ImportError("Cannot initialize required components")

# === TEXT PREPROCESSING ===
class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text[:MAX_TEXT_LENGTH])
        return text.strip()

    @staticmethod
    def extract_vendor_info(text: str) -> Dict[str, str]:
        vendor_info = {}
        gst_pattern = r'GSTIN\s*[:\-]?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})'
        if gst_match := re.search(gst_pattern, text, re.IGNORECASE):
            vendor_info['gst'] = gst_match.group(1)

        date_pattern = r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})'
        if date_match := re.search(date_pattern, text):
            vendor_info['date'] = date_match.group(1)
        return vendor_info

    @staticmethod
    def group_easyocr_text_by_rows(results: List[tuple], y_threshold: int = 10) -> List[List[Dict]]:
        words = [
            {
                'text': result[1],
                'x': result[0][0][0],
                'y': (result[0][0][1] + result[0][2][1]) / 2,
                'confidence': result[2]
            }
            for result in results if result[2] > 0.5
        ]
        
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: (w['y'], w['x']))
        rows, current_row, last_y = [], [], sorted_words[0]['y']

        for word in sorted_words:
            if abs(word['y'] - last_y) <= y_threshold:
                current_row.append(word)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda w: w['x']))
                current_row = [word]
            last_y = word['y']

        if current_row:
            rows.append(sorted(current_row, key=lambda w: w['x']))
        return rows[:30]

    @staticmethod
    def format_table_data(rows: List[List[Dict]]) -> str:
        simplified_rows = [" | ".join(word['text'] for word in row) for row in rows]
        return "\n".join(simplified_rows[:30])

def fix_json_response(json_str: str) -> str:
    cleaned_str = re.sub(r'```json\s*', '', json_str, flags=re.IGNORECASE)
    cleaned_str = re.sub(r'```\s*$', '', cleaned_str)
    json_match = re.search(r'\{.*\}', cleaned_str, re.DOTALL)
    return json_match.group(0).strip() if json_match else ""

async def call_phi3_for_structured_data(raw_text: str, table_text: str) -> Dict:
    if not phi3_model or not phi3_tokenizer:
        raise Exception("Phi3 Mini model not available.")

    if phi3_tokenizer.pad_token_id is None:
        phi3_tokenizer.pad_token_id = phi3_tokenizer.eos_token_id

    prompt = f"""<|system|>
Extract structured invoice data and return ONLY valid JSON.
Rules:
- Return JSON in exact format
- Use null for missing values
- Date format: DD-MM-YYYY
- No explanations
{{
  "Vendor Name": "string",
  "GST No": "string",
  "Date": "string",
  "Items": [
    {{
      "Item Name": "string",
      "HSN Code": "string or null",
      "Qty": number or null,
      "Rate": number or null,
      "MRP": number or null
    }}
  ]
}}
<|user|>
INVOICE: {raw_text[:MAX_TEXT_LENGTH]}
TABLE: {table_text[:500]}
<|end|>"""

    try:
        with torch.no_grad():
            inputs = phi3_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_attention_mask=True
            )
            inputs = inputs.to(device)
            
            outputs = phi3_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=phi3_tokenizer.pad_token_id,
                num_beams=1,
                stopping_criteria=None
            )
            
            response_text = phi3_tokenizer.decode(outputs[0], skip_special_tokens=True)
            fixed_json_str = fix_json_response(response_text.split("<|assistant|>")[-1].strip())
            
            gc.collect()
            return json.loads(fixed_json_str)
    except Exception as e:
        logger.critical(f"Phi3 error: {e}")
        vendor_details = TextPreprocessor.extract_vendor_info(raw_text)
        return {
            "Vendor Name": "Extraction Failed",
            "GST No": vendor_details.get("gst", "N/A"),
            "Date": vendor_details.get("date", "N/A"),
            "Items": [],
            "error": f"AI model extraction failed: {str(e)}",
            "raw_text": raw_text[:MAX_TEXT_LENGTH],
            "table_text": table_text[:500]
        }

async def process_single_image_from_path(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {"status": "error", "file_name": os.path.basename(image_path), "error": "File not found"}

    if not easyocr_reader:
        return {"status": "error", "file_name": os.path.basename(image_path), "error": "EasyOCR not available"}

    logger.critical(f"Processing: {image_path}")
    
    try:
        image = Image.open(image_path).convert('L')  # Grayscale
        image.thumbnail(MAX_IMAGE_SIZE)
        image_array = np.array(image)
        results = await asyncio.to_thread(
            easyocr_reader.readtext,
            image_array,
            detail=1,
            paragraph=False,
            batch_size=1
        )
        
        if not results:
            return {"status": "error", "file_name": os.path.basename(image_path), "message": "No text found"}

        raw_text = " ".join([result[1] for result in results])
        logger.critical(f"Raw text extracted: {raw_text[:MAX_TEXT_LENGTH]}")
        rows = TextPreprocessor.group_easyocr_text_by_rows(results)
        table_text = TextPreprocessor.format_table_data(rows)
        logger.critical(f"Table text: {table_text[:500]}")

        structured_data = await call_phi3_for_structured_data(raw_text, table_text)
        invoice_data = InvoiceData.model_validate(structured_data)

        del image, image_array
        gc.collect()

        return {
            "status": "success",
            "file_name": os.path.basename(image_path),
            "structured_data": invoice_data.model_dump(by_alias=True)
        }
    except ValidationError as e:
        logger.critical(f"Validation failed: {e}")
        return {
            "status": "error",
            "file_name": os.path.basename(image_path),
            "error": "Data validation failed",
            "details": e.errors(),
            "raw_text": raw_text[:MAX_TEXT_LENGTH] if 'raw_text' in locals() else "N/A"
        }
    except Exception as e:
        logger.critical(f"Processing error: {e}")
        return {
            "status": "error",
            "file_name": os.path.basename(image_path),
            "error": str(e),
            "raw_text": raw_text[:MAX_TEXT_LENGTH] if 'raw_text' in locals() else "N/A"
        }

async def main():
    image_path = "/home/thejas/AIBILLO/textile/aibillo-ocr/test.jpg"
    result = await process_single_image_from_path(image_path)
    print(json.dumps(result, indent=2))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())
