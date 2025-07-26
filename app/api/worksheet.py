from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
from io import BytesIO
from PIL import Image

from app.agents.worksheet_agent import WorksheetAgent

router = APIRouter()

worksheet_agent = WorksheetAgent()


def remove_text_references(obj):
    """
    Recursively clean any mention of 'the provided text', 'the text', etc. from strings in dictionaries/lists.
    """
    phrases = [
        "the provided text",
        "the text explicitly defines this",
        "the text describes",
        "as mentioned in the provided text",
        "as mentioned in the text",
        "the text",
        "provided text",
        "Use the provided text to find your answers.",
        "Use the provided text",
        "From the provided text",
        "from the provided text",
        ". The text ",
        ".  The text",
        "as long as they are mentioned in the provided text",
        # Add more phrases if needed
    ]

    # Lowercase matches only
    def clean_string(s):
        for p in phrases:
            s = s.replace(p, "").replace(p.capitalize(), "")
        # Remove stray spaces and dots
        return ' '.join(s.split()).strip(' .')

    if isinstance(obj, dict):
        return {k: remove_text_references(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_text_references(v) for v in obj]
    elif isinstance(obj, str):
        return clean_string(obj)
    return obj


@router.post("/generate")
async def generate_worksheet(
        images: List[UploadFile] = File(..., description="Upload one or more image files (jpg, jpeg, png, etc.)"),
        grade_level: str = Form("5"),
        difficulty: str = Form("medium"),
        mcq_count: int = Form(4),
        short_answer_count: int = Form(3),
        fill_blank_count: int = Form(2),
        true_false_count: int = Form(1)
):
    """
    Generate a worksheet based on uploaded textbook image files (jpg, jpeg, png, etc.).
    """
    try:
        pil_images = []
        for image_file in images:
            try:
                contents = await image_file.read()
                img = Image.open(BytesIO(contents))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pil_images.append(img)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image upload: {image_file.filename}: {str(e)}")

        result = worksheet_agent.generate_worksheet(
            images=pil_images,
            grade_level=grade_level,
            difficulty=difficulty,
            mcq_count=mcq_count,
            short_answer_count=short_answer_count,
            fill_blank_count=fill_blank_count,
            true_false_count=true_false_count
        )

        clean_worksheet = remove_text_references(result["worksheet"]) if result.get("worksheet") else None

        response = {
            "success": clean_worksheet is not None and result.get("error") is None,
            "worksheet": clean_worksheet,
            "parameters": {
                "grade_level": grade_level,
                "difficulty": difficulty,
                "question_distribution": {
                    "multiple_choice": mcq_count,
                    "short_answer": short_answer_count,
                    "fill_in_blank": fill_blank_count,
                    "true_false": true_false_count
                },
                "images_processed": len(pil_images)
            },
            "retries": result.get("retries", []),
            "succeeded_attempt": result.get("succeeded_attempt"),
            "raw_content": result.get("raw_content")
        }
        if result.get("error"):
            response["error"] = result["error"]
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
