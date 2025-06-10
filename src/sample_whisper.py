#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Speech-to-Text API Backend
Поддержка русского языка с возможностью расширения для казахского
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Optional
import wave
import ssl
import ffmpeg
import io
import soundfile as sf
import base64

# Отключаем проверку SSL сертификатов
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchaudio
import whisper
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели данных
class TranscriptionRequest(BaseModel):
    audio_data: str  # base64 encoded audio
    language: str = "ru"
    model_size: str = "base"

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    confidence: float
    processing_time: float

class ConnectionManager:
    """Менеджер WebSocket соединений для реального времени"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_configs: Dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket, config: dict = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_configs[websocket] = config or {"language": "ru", "model": "base"}
        logger.info(f"Новое WebSocket соединение. Всего активных: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.client_configs.pop(websocket, None)
            logger.info(f"WebSocket соединение закрыто. Активных: {len(self.active_connections)}")
    
    async def send_transcription(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Ошибка отправки данных через WebSocket: {e}")
            self.disconnect(websocket)

class STTProcessor:
    """Класс для обработки речи в текст"""
    
    def __init__(self):
        self.models = {}
        self.supported_languages = {
            "ru": "russian",
            "kz": "kazakh",  # Для будущей поддержки казахского
            "en": "english"
        }
        self.load_models()
    
    def load_models(self):
        """Загрузка STT моделей"""
        try:
            # Загружаем базовую модель Whisper для русского языка
            logger.info("Загрузка модели Whisper...")
            
            # Проверяем доступность CUDA
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Используется устройство: {device}")
            
            # Загружаем разные размеры моделей
            # ["tiny", "base", "small", "medium", "large"]
            model_sizes = ["medium"]
            
            for size in model_sizes:
                try:
                    model = whisper.load_model(size, device=device)
                    self.models[size] = model
                    logger.info(f"Модель {size} загружена успешно")
                except Exception as e:
                    logger.error(f"Ошибка загрузки модели {size}: {e}")
            
            if not self.models:
                raise Exception("Не удалось загрузить ни одной модели")
                
        except Exception as e:
            logger.error(f"Критическая ошибка при загрузке моделей: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Предобработка аудио данных"""
        # Конвертируем в float32 и нормализуем
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Нормализация амплитуды
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    
    def load_audio_file(self, file_path: str) -> tuple:
        """Загрузка аудио файла с помощью librosa"""
        try:
            # Загружаем аудио с автоматической конвертацией в 16kHz mono
            audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Ошибка загрузки аудио файла: {e}")
            raise
    
    def process_audio_bytes(self, audio_bytes: bytes) -> tuple:
        """Обработка аудио из байтов с поддержкой разных форматов"""
        try:
            # Пробуем сначала через soundfile
            try:
                with io.BytesIO(audio_bytes) as buf:
                    audio_data, sample_rate = sf.read(buf, dtype='float32')
                    logger.info(f"Soundfile прочитал аудио: {audio_data.shape} {sample_rate}")
            except Exception as e:
                logger.warning(f"Soundfile не смог прочитать аудио: {e}")
                
                # Пробуем через librosa
                try:
                    with io.BytesIO(audio_bytes) as buf:
                        audio_data, sample_rate = librosa.load(buf, sr=16000, mono=True)
                except Exception as e:
                    logger.warning(f"Librosa не смог прочитать аудио: {e}")
                    
                    # Пробуем через torchaudio
                    try:
                        with io.BytesIO(audio_bytes) as buf:
                            audio_data, sample_rate = torchaudio.load(buf)
                            audio_data = audio_data.numpy()
                            if audio_data.ndim > 1:
                                audio_data = audio_data.mean(axis=0)
                    except Exception as e:
                        logger.error(f"Не удалось прочитать аудио ни одним из методов: {e}")
                        raise ValueError("Неподдерживаемый формат аудио")

            # Приводим к 16kHz и моно
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Нормализация
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data, sample_rate
        
        except Exception as e:
            logger.error(f"Ошибка обработки аудио байтов: {e}")
            raise
    
    async def transcribe_audio(self, audio_bytes: bytes, language: str = "ru", model_size: str = "base") -> Dict:
        """Транскрипция аудио из байтового буфера"""
        start_time = time.time()
        try:
            # Выбираем модель
            model = self.models.get(model_size, self.models.get("base"))
            if not model:
                raise ValueError(f"Модель {model_size} недоступна")

            # Получаем numpy-массив аудио
            audio_data, sample_rate = self.process_audio_bytes(audio_bytes)
            audio_data = self.preprocess_audio(audio_data, sample_rate)

            # Whisper ожидает 16kHz float32 numpy array
            result = model.transcribe(
                audio=audio_data,
                language=language,
                task="transcribe",
                fp16=torch.cuda.is_available(),
                temperature=0.0,
                best_of=5,
                beam_size=5,
                patience=1.0,
                length_penalty=1.0,
                suppress_tokens=[-1],
                initial_prompt="Это аудио на русском языке."
            )

            processing_time = time.time() - start_time

            res = result["text"].strip()
            logger.info(f"Транскрипция завершена: {res}")
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "confidence": self._calculate_confidence(result),
                "processing_time": processing_time,
                "segments": result.get("segments", [])
            }
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _calculate_confidence(self, result: dict) -> float:
        """Расчет уверенности на основе сегментов"""
        if "segments" not in result or not result["segments"]:
            return 0.5  # Средняя уверенность по умолчанию
        
        total_confidence = 0
        total_length = 0
        
        for segment in result["segments"]:
            if "avg_logprob" in segment:
                # Конвертируем логарифмическую вероятность в уверенность
                confidence = np.exp(segment["avg_logprob"])
                length = len(segment.get("text", ""))
                total_confidence += confidence * length
                total_length += length
        
        return total_confidence / total_length if total_length > 0 else 0.5

# Инициализация компонентов
app = FastAPI(
    title="Real-time STT API",
    description="API для транскрипции речи в реальном времени с поддержкой русского языка",
    version="1.0.0"
)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные объекты
connection_manager = ConnectionManager()
stt_processor = STTProcessor()

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    logger.info("Запуск STT API сервера...")
    logger.info(f"Доступные модели: {list(stt_processor.models.keys())}")

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Real-time STT API",
        "status": "running",
        "supported_languages": list(stt_processor.supported_languages.keys()),
        "available_models": list(stt_processor.models.keys())
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "models_loaded": len(stt_processor.models),
        "active_connections": len(connection_manager.active_connections)
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = "ru",
    model_size: str = "base"
):
    """HTTP endpoint для транскрипции загруженного файла"""
    
    if language not in stt_processor.supported_languages:
        raise HTTPException(
            status_code=400, 
            detail=f"Язык {language} не поддерживается. Доступные: {list(stt_processor.supported_languages.keys())}"
        )
    
    if model_size not in stt_processor.models:
        raise HTTPException(
            status_code=400,
            detail=f"Модель {model_size} недоступна. Доступные: {list(stt_processor.models.keys())}"
        )
    
    try:
        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Транскрибируем
        result = await stt_processor.transcribe_audio(
            content, language, model_size
        )
        
        # Удаляем временный файл
        os.unlink(tmp_file_path)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return TranscriptionResponse(**result)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint для транскрипции в реальном времени"""
    
    await connection_manager.connect(websocket, {"language": "ru", "model": "base"})
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "config":
                connection_manager.client_configs[websocket].update(message.get("config", {}))
                await connection_manager.send_transcription(websocket, {
                    "type": "config_updated",
                    "config": connection_manager.client_configs[websocket]
                })
                
            elif message.get("type") == "audio":
                try:
                    audio_b64 = message.get("audio_data", "")
                    if not audio_b64:
                        raise ValueError("Пустые аудио данные")
                        
                    # Декодируем base64 аудио
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception as e:
                        raise ValueError(f"Ошибка декодирования base64: {e}")
                    
                    # Получаем параметры из конфига клиента
                    config = connection_manager.client_configs[websocket]
                    language = message.get("language", config.get("language", "ru"))
                    model_size = message.get("model_size", config.get("model", "base"))
                    
                    # Транскрибируем
                    result = await stt_processor.transcribe_audio(
                        audio_bytes, language, model_size
                    )
                    
                    # Отправляем результат
                    await connection_manager.send_transcription(websocket, {
                        "type": "transcription",
                        "result": result
                    })
                    
                except Exception as e:
                    logger.error(f"Ошибка обработки аудио через WebSocket: {e}")
                    await connection_manager.send_transcription(websocket, {
                        "type": "error",
                        "message": str(e)
                    })
    
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket ошибка: {e}")
        connection_manager.disconnect(websocket)

@app.get("/models")
async def get_available_models():
    """Получить список доступных моделей"""
    return {
        "models": list(stt_processor.models.keys()),
        "languages": stt_processor.supported_languages,
        "default_language": "ru",
        "default_model": "base"
    }

if __name__ == "__main__":
    # Настройки для запуска
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Запуск сервера на {host}:{port}")
    
    uvicorn.run(
        "sample_whisper:app",
        host=host,
        port=port,
        reload=False,  # Отключаем reload в продакшене
        workers=1,     # Один воркер для экономии памяти с ML моделями
        log_level="info"
    )