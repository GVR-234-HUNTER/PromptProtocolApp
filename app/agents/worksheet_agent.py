import io
import logging
import os
import re
from typing import List, Dict, Any

import dotenv
import requests
from PIL import Image

# Load environment variables from .env file
dotenv.load_dotenv()


def _group_questions_by_type(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Group questions by their type and create a separate student worksheet and answer key
    """
    # Student worksheet - questions only (no answers)
    student_questions = {
        "multiple_choice": [],
        "short_answer": [],
        "fill_in_blank": [],
        "true_false": []
    }

    # Teacher answer key - answers only
    answer_key = {
        "multiple_choice": [],
        "short_answer": [],
        "fill_in_blank": [],
        "true_false": []
    }

    for question in questions:
        question_type = question.get("type", "short_answer")

        if question_type in student_questions:
            # For students: Remove answers and explanations
            student_question = {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "options": question.get("options", [])
            }
            student_questions[question_type].append(student_question)

            # For teachers: Just ID, correct answer, and explanation
            teacher_answer = {
                "id": question["id"],
                "correct_answer": question["correct_answer"],
                "explanation": question["explanation"]
            }
            answer_key[question_type].append(teacher_answer)

    # Add counts for each type
    question_counts = {
        "multiple_choice_count": len(student_questions["multiple_choice"]),
        "short_answer_count": len(student_questions["short_answer"]),
        "fill_in_blank_count": len(student_questions["fill_in_blank"]),
        "true_false_count": len(student_questions["true_false"])
    }

    return {
        "student_worksheet": student_questions,
        "teacher_answer_key": answer_key,
        "question_counts": question_counts
    }


def _self_evaluate_worksheet(worksheet, question_targets, must_have_answers=True):
    """
    Evaluate if the worksheet matches required criteria and is sensible. Returns (bool, str).
    """
    if not worksheet or "question_counts" not in worksheet:
        return False, "Parsing failed or worksheet missing question_counts."
    qc = worksheet["question_counts"]
    for key, count in question_targets.items():
        k = key + "_count"  # keys: 'multiple_choice_count', etc.
        if qc.get(k, 0) != count:
            return False, f"Expected {count} {key.replace('_', ' ')}; found {qc.get(k, 0)}."
    # Optionally ensure every answer/explanation exists for the teacher key
    if must_have_answers:
        tk = worksheet.get("teacher_answer_key") or {}
        for qtype, lst in tk.items():
            for item in lst:
                if "correct_answer" not in item or not str(item["correct_answer"]).strip():
                    return False, f"Missing answer for {qtype} question ID {item.get('id')}"
                if "explanation" not in item:
                    return False, f"Missing explanation for {qtype} question ID {item.get('id')}"
    return True, "OK"


class WorksheetAgent:
    def __init__(self, gemini_model="gemini-1.5-flash"):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        self.gemini_url = (
            f"https://generativelanguage.googleapis.com/v1/models/{gemini_model}:generateContent"
            f"?key={self.gemini_api_key}"
        )
        self.state = {}

    def _call_gemini(self, prompt, images=None):
        """
        Call Gemini API with text prompt and optional images
        """
        # Prepare parts with text and images
        parts = [{"text": prompt}]

        # Add images if provided
        if images and len(images) > 0:
            for img in images:
                if isinstance(img, Image.Image):
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    # Encode as base64
                    import base64
                    encoded_img = base64.b64encode(img_bytes).decode('utf-8')

                    # Add to parts
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": encoded_img
                        }
                    })

        payload = {"contents": [{"parts": parts}]}
        response = requests.post(
            self.gemini_url, json=payload, headers={"Content-Type": "application/json"}, timeout=60
        )
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API failed: {response.status_code}: {response.text}")
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']

    def _parse_worksheet_to_json(self, worksheet_text: str) -> Dict[str, Any]:
        """
        Parse the AI-generated worksheet text into structured JSON format with grouped questions
        """
        try:
            lines = worksheet_text.split('\n')

            # Extract metadata
            title = ""
            grade_level = ""
            subject = ""
            difficulty = ""
            instructions = ""

            # Find title and metadata
            for line in lines:
                if line.startswith("**WORKSHEET:"):
                    title = line.replace("**WORKSHEET:", "").replace("**", "").strip()
                elif line.startswith("**Grade Level:**"):
                    grade_level = line.replace("**Grade Level:**", "").strip()
                elif line.startswith("**Subject:**"):
                    subject = line.replace("**Subject:**", "").strip()
                elif line.startswith("**Difficulty:**"):
                    difficulty = line.replace("**Difficulty:**", "").strip()

            # Extract instructions
            instruction_start = False
            for line in lines:
                if "**Instructions for Students:**" in line:
                    instruction_start = True
                    continue
                elif instruction_start and line.startswith("**QUESTIONS:**"):
                    break
                elif instruction_start and line.strip():
                    instructions += line.strip() + " "

            # Extract questions and answers
            questions = []

            # Find the questions section
            questions_section = []
            answers_section = []
            current_section = None

            for line in lines:
                if "**QUESTIONS:**" in line:
                    current_section = "questions"
                    continue
                elif "**ANSWER KEY:**" in line:
                    current_section = "answers"
                    continue
                elif current_section == "questions" and line.strip():
                    questions_section.append(line)
                elif current_section == "answers" and line.strip():
                    answers_section.append(line)

            # Parse questions
            question_number = 1
            current_question = None

            for line in questions_section:
                line = line.strip()
                if not line:
                    continue

                # Check if it's a new question (starts with number)
                if re.match(r'^\d+\.', line):
                    if current_question:
                        questions.append(current_question)

                    # Extract question type and text
                    question_text = re.sub(r'^\d+\.\s*', '', line)
                    question_type = "short_answer"  # default

                    if "**Multiple Choice:**" in question_text:
                        question_type = "multiple_choice"
                        question_text = question_text.replace("**Multiple Choice:**", "").strip()
                    elif "**Fill-in-the-blank:**" in question_text:
                        question_type = "fill_in_blank"
                        question_text = question_text.replace("**Fill-in-the-blank:**", "").strip()
                    elif "**True or False:**" in question_text:
                        question_type = "true_false"
                        question_text = question_text.replace("**True or False:**", "").strip()
                    elif "**Short Answer:**" in question_text:
                        question_type = "short_answer"
                        question_text = question_text.replace("**Short Answer:**", "").strip()

                    current_question = {
                        "id": question_number,
                        "type": question_type,
                        "question": question_text,
                        "options": [],
                        "correct_answer": "",
                        "explanation": ""
                    }
                    question_number += 1

                # Check if it's an option (a), b), c), d))
                elif current_question and re.match(r'^\s*[a-d]\)', line):
                    option_text = re.sub(r'^\s*[a-d]\)\s*', '', line)
                    current_question["options"].append(option_text)

            # Add the last question
            if current_question:
                questions.append(current_question)

            # Parse answers
            answer_number = 1
            for line in answers_section:
                line = line.strip()
                if not line:
                    continue

                if re.match(r'^\d+\.', line):
                    answer_text = re.sub(r'^\d+\.\s*', '', line)

                    # Find corresponding question and add answer
                    if answer_number <= len(questions):
                        # Extract the correct answer and explanation
                        if questions[answer_number - 1]["type"] == "multiple_choice":
                            # Extract the letter answer (a, b, c, d)
                            match = re.search(r'\*\*([a-d])\)', answer_text)
                            if match:
                                questions[answer_number - 1]["correct_answer"] = match.group(1)

                            # Extract explanation after **
                            explanation_match = re.search(r'\*\*.*?\*\*\s*(.*)', answer_text)
                            if explanation_match:
                                questions[answer_number - 1]["explanation"] = explanation_match.group(1)
                        else:
                            # For other question types, the whole answer is the correct answer
                            if "**" in answer_text:
                                parts = answer_text.split("**")
                                if len(parts) >= 2:
                                    questions[answer_number - 1]["correct_answer"] = parts[1].strip()
                                    if len(parts) > 2:
                                        questions[answer_number - 1]["explanation"] = parts[2].strip()
                            else:
                                questions[answer_number - 1]["correct_answer"] = answer_text

                    answer_number += 1

            # Group questions by type
            grouped_data = _group_questions_by_type(questions)

            return {
                "worksheet_info": {
                    "title": title,
                    "grade_level": grade_level,
                    "subject": subject,
                    "difficulty": difficulty,
                    "instructions": instructions.strip()
                },
                **grouped_data  # This includes student_worksheet, teacher_answer_key, question_counts
            }

        except Exception as e:
            logging.exception("Error parsing worksheet")
            return {
                "worksheet_info": {
                    "title": "Generated Worksheet",
                    "grade_level": "Unknown",
                    "subject": "General",
                    "difficulty": "Medium",
                    "instructions": "Complete all questions to the best of your ability."
                },
                "student_worksheet": {
                    "multiple_choice": [],
                    "short_answer": [],
                    "fill_in_blank": [],
                    "true_false": []
                },
                "teacher_answer_key": {
                    "multiple_choice": [],
                    "short_answer": [],
                    "fill_in_blank": [],
                    "true_false": []
                },
                "question_counts": {
                    "multiple_choice_count": 0,
                    "short_answer_count": 0,
                    "fill_in_blank_count": 0,
                    "true_false_count": 0
                },
                "error": f"Parsing error: {str(e)}"
            }

    def generate_worksheet(self, images, grade_level="5", subject="General",
                        difficulty="medium", question_count=10, mcq_count=4,
                           short_answer_count=3, fill_blank_count=2, true_false_count=1, max_attempts=3):
        """
        Generate a worksheet based on textbook images, with agentic retry/self-evaluation loop.
        """
        result = {"worksheet": None, "raw_content": None, "error": None, "retries": []}

        if not images or len(images) == 0:
            result["error"] = "At least one image is required."
            return result

        # Validate total question count matches breakdown
        total_specified = mcq_count + short_answer_count + fill_blank_count + true_false_count
        if total_specified != question_count:
            question_count = total_specified

        question_targets = {
            "multiple_choice": mcq_count,
            "short_answer": short_answer_count,
            "fill_in_blank": fill_blank_count,
            "true_false": true_false_count
        }

        last_reason = None
        for attempt in range(1, max_attempts + 1):
            # Build prompt
            images_text = "images" if len(images) > 1 else "image"
            pages_text = f"across all {len(images)} pages" if len(images) > 1 else "from this page"
            prompt = f"""
            Analyze these {len(images)} textbook page {images_text} and create a comprehensive worksheet that covers content {pages_text}.

            Requirements:
            - Grade Level: {grade_level}
            - Subject: {subject}
            - Difficulty: {difficulty}
            - Total Questions: EXACTLY {question_count}

            Question Distribution:
            - Multiple Choice: EXACTLY {mcq_count} questions
            - Short Answer: EXACTLY {short_answer_count} questions
            - Fill-in-the-blank: EXACTLY {fill_blank_count} questions
            - True/False: EXACTLY {true_false_count} questions

            IMPORTANT: Follow this EXACT format for consistent parsing:

            **WORKSHEET: [Topic from Images]**
            **Grade Level:** {grade_level}
            **Subject:** {subject}
            **Difficulty:** {difficulty}

            **Instructions for Students:**
            Read each question carefully and answer to the best of your ability. Use the provided text to find your answers.

            **QUESTIONS:**

            [Create EXACTLY {mcq_count} Multiple Choice Questions]
            1. **Multiple Choice:** [Question text based on content from the images]
             a) [Option A]
             b) [Option B]
             c) [Option C]
             d) [Option D]

            [Continue with more MCQ questions...]

            [Create EXACTLY {short_answer_count} Short Answer Questions]
            {mcq_count + 1}. **Short Answer:** [Question text based on content from the images]

            [Continue with more short answer questions...]

            [Create EXACTLY {fill_blank_count} Fill-in-the-blank Questions]
            {mcq_count + short_answer_count + 1}. **Fill-in-the-blank:** [Question with _______ blank based on content]

            [Continue with more fill-in-the-blank questions...]

            [Create EXACTLY {true_false_count} True/False Questions]
            {mcq_count + short_answer_count + fill_blank_count + 1}. **True or False:** [Statement to evaluate based on content]

            [Continue with more true/false questions...]

            **ANSWER KEY:**

            [Provide answers for ALL {question_count} questions in order]
            1. **a)** [Explanation of correct answer]
            2. **b)** [Explanation of correct answer]
            [Continue for all questions...]

            Make sure:
            - Use EXACTLY the format shown above
            - Base all questions on content from ALL the provided textbook images
            - Ensure questions cover material across all pages if multiple images provided
            - Appropriate difficulty for grade {grade_level}
            - Clear answer key with explanations
            - Maintain the exact question count distribution specified
            """
            try:
                raw_content = self._call_gemini(prompt, images)
                if not raw_content or not raw_content.strip():
                    last_reason = "Failed to generate worksheet content"
                    result["retries"].append(last_reason)
                    continue
                structured_worksheet = self._parse_worksheet_to_json(raw_content)
                passed, reason = _self_evaluate_worksheet(
                    structured_worksheet, question_targets, must_have_answers=True
                )
                if passed:
                    # Success on this attempt
                    self.state["last_worksheet"] = structured_worksheet
                    self.state["last_raw_content"] = raw_content
                    result["worksheet"] = structured_worksheet
                    result["raw_content"] = raw_content
                    result["succeeded_attempt"] = attempt
                    result["retries"] = result["retries"]
                    return result
                else:
                    last_reason = reason
                    result["retries"].append(f"Attempt {attempt}: {reason}")
            except Exception as exc:
                last_reason = str(exc)
                result["retries"].append(f"Attempt {attempt} exception: {last_reason}")
        # If the loop ends without success
        result["error"] = f"Worksheet generation failed after {max_attempts} attempts. Last reason: {last_reason}"
        return result


