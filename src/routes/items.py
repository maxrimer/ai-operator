from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(tags=["Items"])

class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

@router.get("/items", response_model=List[Item])
async def get_items():
    """
    Get all items
    """
    # Example data - replace with actual database operations
    return [
        Item(id=1, name="Item 1", description="Description 1"),
        Item(id=2, name="Item 2", description="Description 2")
    ]

@router.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    """
    Get item by ID
    """
    # Example data - replace with actual database operations
    if item_id == 1:
        return Item(id=1, name="Item 1", description="Description 1")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Item with id {item_id} not found"
    )

@router.post("/items", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    """
    Create a new item
    """
    # Example implementation - replace with actual database operations
    return Item(
        id=3,  # In real implementation, this would be generated
        name=item.name,
        description=item.description
    ) 