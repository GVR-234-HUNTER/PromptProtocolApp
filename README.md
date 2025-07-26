# AI Agent Platform

This platform provides multiple AI agents that perform different tasks:

## Available Agents

### 1. Diagram Agent

Generates diagrams from text descriptions using the Gemini API and Kroki for visualization.

- **Endpoint**: `/api/diagram/generate`
- **Features**:
  - Text-to-diagram generation
  - Multiple diagram formats (graphviz, mermaid, etc.)
  - Different output formats (PDF, SVG, PNG)

### 2. Worksheet Agent

Generates educational worksheets from textbook images using the Gemini API.

- **Endpoint**: `/api/worksheet/generate`
- **Features**:
  - Image-to-worksheet generation
  - Customizable question types and counts
  - Structured output with student worksheet and teacher answer key
  - Multiple difficulty levels

## Agent Architecture

Each agent follows a similar architecture:

1. **Agent Class**: Contains the core logic and maintains state
2. **API Endpoint**: Handles HTTP requests and responses
3. **Models**: Defines the data structures for requests and responses

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r backend/app/requirements.txt`
3. Set up environment variables in a `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. Run the server: `uvicorn backend.app.main:app --reload`

## Usage Examples

### Generate a Diagram

```python
import requests

response = requests.post(
    "http://localhost:8000/api/diagram/generate",
    json={
        "prompt": "A flowchart showing the water cycle",
        "code_style": "graphviz",
        "output_format": "svg"
    }
)
result = response.json()
```

### Generate a Worksheet

```python
import requests
import base64

# Load and encode an image
with open("textbook_page.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/api/worksheet/generate",
    json={
        "images": [encoded_image],
        "grade_level": "8",
        "subject": "Science",
        "difficulty": "medium",
        "question_count": 10,
        "mcq_count": 5,
        "short_answer_count": 2,
        "fill_blank_count": 2,
        "true_false_count": 1
    }
)
result = response.json()
```