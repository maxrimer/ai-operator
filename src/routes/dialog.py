from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Depends, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.repositories.chat_repository import ChatRepository
from src.database import get_db
from src.models.chat import Chat
from src.models.dialog_info import DialogInfo
from loguru import logger


router = APIRouter(tags=["Dialog"], prefix='/dialog')


class DialogRequestDto(BaseModel):
    chat_id: int
    role: str 
    text: str

class DialogHintRequestDto(BaseModel):
    chat_id: int
    dialog_id: int 
    is_used: bool

class DialogResponseDto(BaseModel):
    chat_id: int
    customer_number: Optional[str] = None
    messages: Optional[List] = []
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    summary: Optional[str] = None

def make_dialog_response(chat: Chat) -> DialogResponseDto:
    return DialogResponseDto(
            chat_id=chat.id,
            customer_number=chat.customer_number,
            created_at=chat.created_at.isoformat() if chat.created_at else None,
            updated_at=chat.updated_at.isoformat() if chat.updated_at else None,
            summary=chat.summary,
            messages=[chat.messages[-1]] if chat.messages else [],
            status=chat.status
        )

@router.get("/chats", response_model=List[DialogResponseDto], status_code=status.HTTP_200_OK)
async def get_chats(db: Session = Depends(get_db)):
    """ Get all chats information """

    chat_repository = ChatRepository(db)
    chats = chat_repository.get_chats()
    
    return [
        make_dialog_response(chat=chat)
        for chat in chats
    ]

@router.post("/create", response_model=DialogResponseDto, status_code=status.HTTP_201_CREATED)
async def create_chat(customer_number: str, db: Session = Depends(get_db)):
    """ Create a new chat in the database """

    new_chat = Chat()
    new_chat.customer_number = customer_number
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return make_dialog_response(chat=new_chat)


@router.get("/{chat_id}", response_model=DialogResponseDto, status_code=status.HTTP_200_OK)
async def get_chat(chat_id: int, db: Session = Depends(get_db)):
    """ Get chat information by chat ID """

    chat_repository = ChatRepository(db)
    chat : Chat = chat_repository.get_chat_by_id(chat_id)
    
    return make_dialog_response(chat=chat)


@router.post("", response_model=DialogResponseDto)
async def pipline_run(req_dto: DialogRequestDto, db: Session = Depends(get_db)):
    """
    """
    chat_repository = ChatRepository(db)
    chat = chat_repository.get_chat_by_id(chat_id=req_dto.chat_id)

    logger.info(f"Текущие сообщения: {chat.messages}")
            
    # Получаем текущие сообщения или инициализируем пустой список
    current_messages = chat.messages or []
    
    # Проверяем, что current_messages - это список
    if not isinstance(current_messages, list):
        logger.warning(f"messages не является списком: {current_messages}")
        current_messages = []
    
    # Генерируем dialog_id
    last_dialog_id = current_messages[-1]['dialog_id'] if current_messages else 0
    new_message=DialogInfo(
        dialog_id=last_dialog_id+1,
        role=req_dto.role,
        text=req_dto.text
    )
    # TODO: проверка на role = client
    # и запуск pipeline
    # all_text = ' '.join([x['text'] for x in current_messages])

    
    # TODO: new_message.hint_type = 'q/a'
    
    # Создаем новый список, чтобы SQLAlchemy заметил изменение
    updated_messages = current_messages + [new_message.to_dict()]
    
    # TODO: добавить суфлерский хинт
    # suffler_message = DialogInfo(
    #     dialog_id=new_message.dialog_id+1,
    #     role='suffler',
    #     text='hint'
    # )
    # updated_messages = current_messages + [suffler_message.to_dict()]
    
    # Обновляем столбец messages
    chat.messages = updated_messages
    
    logger.info(f"Обновленные сообщения: {chat.messages}")
    chat = chat_repository.update_chat(chat=chat)
    logger.info("Изменения успешно сохранены")
    
    return make_dialog_response(chat=chat)

@router.post("/hint", response_model=DialogResponseDto)
async def pipline_run(req_dto: DialogHintRequestDto, db: Session = Depends(get_db)):
    """
    """
    chat_repository = ChatRepository(db)
    chat = chat_repository.get_chat_by_id(chat_id=req_dto.chat_id)
            
    # Получаем текущие сообщения или инициализируем пустой список
    current_messages = chat.messages or []
    
    # Проверяем, что current_messages - это список
    if not isinstance(current_messages, list):
        logger.warning(f"messages не является списком: {current_messages}")
        current_messages = []
    
    # Генерируем dialog_id
    updated_messages = [msg.copy() if msg['dialog_id'] != req_dto.dialog_id 
                                    else {**msg, 'is_used': req_dto.is_used} 
                        for msg in current_messages]
    chat.messages = updated_messages

    chat_repository.update_chat(chat=chat)
    logger.info("Изменения успешно сохранены")
    
    return Response(status_code=200)

@router.post("/close")
async def pipline_run(chat_id: int, db: Session = Depends(get_db)):
    """
    """
    chat_repository = ChatRepository(db)
    chat = chat_repository.get_chat_by_id(chat_id=chat_id)

    chat.status = 'closed'

    chat_repository.update_chat(chat=chat)
    logger.info("Status updated to CLOSE")
    
    return Response(status_code=200)
