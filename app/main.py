from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Import routers
from app.api import diagram, worksheet

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Sahayak Teaching Platform", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, in production you should restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(diagram.router, prefix="/api/diagram", tags=["Diagram Agent"])
app.include_router(worksheet.router, prefix="/api/worksheet", tags=["Worksheet Agent"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Sahayak Teaching Platform", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
