from sqlalchemy import Column, Integer, JSON, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from src.database import Base

class Chat(Base):
    """
    Chat model representing a conversation
    """
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    messages = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    summary = Column(String, nullable=True)
    status = Column(String, nullable=False, default='active')
    customer_number = Column(String, nullable=True)

    def __repr__(self):
        return f"<Chat(id={self.id})>" 