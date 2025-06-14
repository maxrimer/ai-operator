from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Depends, Response
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agent.langgraph_agent import CallState, flow
from src.repositories.chat_repository import ChatRepository
from src.database import get_db
from src.models.chat import Chat
from src.models.dialog_info import DialogInfo
from src.retriever.csv_retriever import s3_client

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

class ChatsResponseDto(BaseModel):
    chat_id: int
    customer_number: Optional[str] = None
    last_message: Optional[str] = ""
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    summary: Optional[str] = None

def make_dialog_response(chat: Chat, one_message: bool = False) -> DialogResponseDto:
    return DialogResponseDto(
            chat_id=chat.id,
            customer_number=chat.customer_number,
            created_at=chat.created_at.isoformat() if chat.created_at else None,
            updated_at=chat.updated_at.isoformat() if chat.updated_at else None,
            summary=chat.summary,
            messages=[chat.messages[-1]] if chat.messages and one_message 
                        else chat.messages if chat.messages and not one_message
                        else [],
            status=chat.status
        )

@router.get("/chats", response_model=List[ChatsResponseDto], status_code=status.HTTP_200_OK)
async def get_chats(db: Session = Depends(get_db)):
    """ Get all chats information """

    chat_repository = ChatRepository(db)
    chats = chat_repository.get_chats()
    
    return [
        ChatsResponseDto(
            chat_id=chat.id,
            customer_number=chat.customer_number,
            created_at=chat.created_at.isoformat() if chat.created_at else None,
            updated_at=chat.updated_at.isoformat() if chat.updated_at else None,
            summary=chat.summary,
            last_message=chat.messages[-1]['text'] if chat.messages else "",
            status=chat.status
        )
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
async def pipline_run(req_dtos: List[DialogRequestDto], db: Session = Depends(get_db)):
    """
    """
    if len(req_dtos) == 0:
        raise HTTPException(status_code=400, detail='len is eq 0')
    
    chat_repository = ChatRepository(db)
    chat = chat_repository.get_chat_by_id(chat_id=req_dtos[0].chat_id)
    logger.info(f"Текущии chat_id: {chat.id}")
    
    # Получаем текущие сообщения или инициализируем пустой список
    current_messages = chat.messages or []

    # Проверяем, что current_messages - это список
    if not isinstance(current_messages, list):
        logger.warning(f"messages не является списком: {current_messages}")
        current_messages = []

    updated_messages = current_messages
    for req_dto in req_dtos:
        
        # Генерируем dialog_id
        last_dialog_id = current_messages[-1]['dialog_id'] if current_messages else 0
        new_message=DialogInfo(
            dialog_id=last_dialog_id+1,
            role=req_dto.role,
            text=req_dto.text
        )
        # Создаем новый список, чтобы SQLAlchemy заметил изменение
        updated_messages = updated_messages + [new_message.to_dict()]

    try:
        
        if len(req_dtos) > 1:
            all_text = '. '.join([x['text'] for x in updated_messages if x['role'] != 'suffler'])
        else:
            # TODO ??? Нужно ли брать предыдушие сообщения ???
            all_text = f' {new_message.text}'
        
        logger.info(f'{len(req_dtos)}: {all_text}')
        
        config = {'configurable': {'thread_id': chat.id}}
        init_state = CallState(customer_query=all_text, customer_id=int(chat.customer_number.replace(" ", "")))
        result = flow.invoke(input=init_state, config=config)
        
        suffler_message = DialogInfo(
            dialog_id=new_message.dialog_id+1,
            role='suffler',
            text=result['hint'],
            hint_type='quetion' if result['is_query_need_clarification'] else 'not quetion',
            confidence=result['confidence'],
            source_name=result['source'],
            source=f'http://94.131.80.240:8000/dialog/download?filename=1'
        )
        updated_messages = updated_messages + [suffler_message.to_dict()]

        # Обновляем столбец messages
        chat.messages = updated_messages
        
        chat = chat_repository.update_chat(chat=chat)
        logger.info("Изменения успешно сохранены")

    except Exception as e:
        logger.error(f'ERROR: {str(e)}')
        return HTTPException(
            status_code=500,
            detail=str(e)
        )
    
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
 

@router.get("/download", status_code=status.HTTP_200_OK)
async def download_file(filename: int):
    """ Download file by filename """
    
    file_path = 'AI-суфлер общий доступ/КРБ/База знаний/Автокредит+на+приобретение+авто+с+пробегом.docx'
    file_ext = file_path.split('.')[-1]

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    
    temp_file = f'temp.{file_ext}'
    logger.info(temp_file)
    s3_client.download_file(file_path, temp_file)
    
    try:
        with open(temp_file, 'rb') as file:
            file_content = file.read()
            
        # Clean up the temporary file
        import os
        os.remove(temp_file)
        
        # Determine content type based on file extension
        content_type = {
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'csv': 'text/csv',
            'pdf': 'application/pdf'
        }.get(file_ext.lower(), 'application/octet-stream')
        
        # Get original filename from path
        original_filename = file_path.split('/')[-1].replace('+', ' ')
        
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                'Content-Disposition': f'attachment; filename="{original_filename}"'
            }
        )
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading file: {str(e)}"
        )
