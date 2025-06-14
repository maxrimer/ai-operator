from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from src.models.chat import Chat



class ChatRepository:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def get_chat_by_id(self, chat_id: int) -> Optional[Chat]:
        """
        Retrieve chat information by chat ID
        """
        query = select(Chat).where(Chat.id == chat_id)
        result = self.db_session.execute(query)
        chat : Chat = result.scalar_one_or_none()
        
        if not chat:
            raise ValueError(f"Chat with ID {chat_id} not found")
            
        return chat
    
    def get_chats(self) -> List[Chat]:
        """
        Retrieve all chats
        """
        query = select(Chat).order_by(desc(Chat.id))
        result = self.db_session.execute(query)
        chats = result.scalars().all()
            
        return chats
    
    def update_chat(self, chat: Chat):
        # Коммитим изменения
        self.db_session.commit()
        self.db_session.refresh(chat)
        
        return chat
            