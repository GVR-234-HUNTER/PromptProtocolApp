import os
import requests
import base64
import ast
import re
import logging
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

class DiagramAgent:
    def __init__(self, gemini_model="gemini-2.5-flash"):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        self.gemini_url = (
            f"https://generativelanguage.googleapis.com/v1/models/{gemini_model}:generateContent"
            f"?key={self.gemini_api_key}"
        )
        self.kroki_base_url = "https://kroki.io"
        self.state = {}

    def _call_gemini(self, prompt):
        payload = { "contents": [ {"parts": [{"text": prompt}]} ]}
        response = requests.post(
            self.gemini_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30
        )
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API failed: {response.status_code}: {response.text}")
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']

    def _clean_generated_code(self, raw_code):
        """Clean Gemini output by handling escapes, removing code block markers, etc."""
        # Replace escaped newlines
        code = raw_code.replace("\\n", "\n")

        # Optional: decode full escape sequences safely
        try:
            code = ast.literal_eval(f"'''{code}'''")
        except Exception as e:
            logging.warning(f"Failed to decode escape sequences: {e}")

        # Remove fenced code block markers if present
        fenced_block_match = re.search(r"```(?:[a-zA-Z0-9]*)?\n(.*?)```", code, flags=re.DOTALL)
        if fenced_block_match:
            code = fenced_block_match.group(1).strip()

        logging.debug(f"Cleaned diagram code:\n{code}")
        return code

    def _render_diagram(self, code, code_style="graphviz", output_format="pdf"):
        if not code or not code.strip():
            raise RuntimeError("No diagram code provided to render.")
        url = f"{self.kroki_base_url}/{code_style}/{output_format}"
        headers = {"Content-Type": "text/plain", "Accept": f"image/{output_format}"}
        response = requests.post(url, data=code.encode("utf-8"), headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Kroki rendering failed with status {response.status_code}: {response.content[:200]!r}"
            )
        encoded_img = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/{output_format};base64,{encoded_img}"

    def _self_evaluate_diagram(self, code, image_uri, code_style, output_format):
        """
        Agentic check: is the code non-empty and did rendering yield a plausible image (not blank, not error)?
        """
        if not code or not code.strip():
            return False, "Code is empty or blank."
        if not image_uri or not image_uri.startswith(f"data:image/{output_format}"):
            return False, "Rendered image URI is missing or incorrect."
        if len(image_uri) < 100:
            return False, "Image data URI is suspiciously short."
        # You could decode base64 and check for certain error-strings or patterns for more robustness.
        return True, "OK"

    def generate_diagram(self, user_prompt, code_style="graphviz", output_format="pdf", max_attempts=3):
        result = {"image": None, "diagram_code": None, "error": None, "retries": []}
        if not user_prompt.strip():
            result["error"] = "Prompt cannot be empty."
            return result
        last_reason = None
        for attempt in range(1, max_attempts + 1):
            try:
                gemini_prompt = (
                    f"Generate a diagram in {code_style} syntax for the following description:\n{user_prompt}"
                )
                raw_code = self._call_gemini(gemini_prompt)
                code = self._clean_generated_code(raw_code)
                if not code.strip():
                    last_reason = "Generated diagram code is empty after cleaning."
                    result["retries"].append(f"Attempt {attempt}: {last_reason}")
                    continue
                # Try rendering
                try:
                    image_data_uri = self._render_diagram(code, code_style, output_format)
                except Exception as render_exc:
                    last_reason = f"Render failed: {render_exc}"
                    result["retries"].append(f"Attempt {attempt}: {last_reason}")
                    continue
                passed, reason = self._self_evaluate_diagram(code, image_data_uri, code_style, output_format)
                if passed:
                    result["diagram_code"] = code
                    result["image"] = image_data_uri
                    result["succeeded_attempt"] = attempt
                    result["retries"] = result["retries"]
                    self.state["last_diagram_code"] = code
                    self.state["last_prompt"] = user_prompt
                    return result
                else:
                    last_reason = reason
                    result["retries"].append(f"Attempt {attempt}: {reason}")
            except Exception as exc:
                last_reason = str(exc)
                result["retries"].append(f"Attempt {attempt} exception: {last_reason}")
        # Fails all attempts
        result["error"] = f"Diagram generation failed after {max_attempts} attempts. Last reason: {last_reason}"
        return result
