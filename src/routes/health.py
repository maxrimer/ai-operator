from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])

class HealthResponse(BaseModel):
    status: str
    version: str

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    ) 