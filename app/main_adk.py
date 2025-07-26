from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Import original routers (can be kept for backward compatibility)
from app.api import diagram, worksheet

# Import ADK-based routers
from app.api import diagram_adk, worksheet_adk, chatbot_adk

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Sahayak Teaching Platform (ADK Version)", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, in production you should restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include original routers (optional, for backward compatibility)
# app.include_router(diagram.router, prefix="/api/diagram", tags=["Diagram Agent (Legacy)"])
# app.include_router(worksheet.router, prefix="/api/worksheet", tags=["Worksheet Agent (Legacy)"])

# Include ADK-based routers
app.include_router(diagram_adk.router, prefix="/api/diagram", tags=["Diagram Agent"])
app.include_router(worksheet_adk.router, prefix="/api/worksheet", tags=["Worksheet Agent"])
app.include_router(chatbot_adk.router, prefix="/api/chatbot", tags=["Chatbot Agent"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Sahayak Teaching Platform (ADK Version)", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0", "implementation": "ADK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_adk:app", host="0.0.0.0", port=8000, reload=True)