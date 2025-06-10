import sounddevice as sd
import numpy as np
import queue
import threading
import time
import logging
from typing import Literal, Optional
from stt_fw_class import STTFasterWhisperModel

# === Настройки логирования ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Настройки ===
SAMPLE_RATE = 16000  # Частота дискретизации (подходит для STT)
BLOCK_SIZE = 4000    # Длина блока (в сэмплах) — ~0.25 сек
CHANNELS = 1
DEVICE = None        # Использовать устройство по умолчанию
BUFFER_DURATION = 5  # Длительность буфера в секундах
MAX_QUEUE_SIZE = 100  # Максимальный размер очереди

class AudioProcessor:
    def __init__(self, 
                 model_path: str, 
                 compute_type: str = 'int8', 
                 device: str = 'cpu', 
                 target_frequency: int = SAMPLE_RATE):
        """Инициализация аудиопроцессора с STT моделью."""
        try:
            self.stt_model = STTFasterWhisperModel(
                path_to_model=model_path, 
                compute_type=compute_type, 
                device=device,
                target_frequency=target_frequency
            )
            logger.info(f"STT модель загружена: {model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки STT модели: {e}")
            raise
        
        self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.buffer = np.empty((0,), dtype=np.float32)
        self.is_running = False
        self.worker_thread = None
        
    def read_audio_from_array(self, audio_array: np.ndarray, sample_rate: int,
                            mono_conversion_mode: Literal['average', 'alternate'] = 'average'):
        """Конвертация numpy массива в AudioFileData."""
        try:
            # Проверка входных данных
            if audio_array.size == 0:
                raise ValueError("Пустой аудиомассив")
            
            # Если аудио двумерное (стерео), применяем моно-конвертацию
            if audio_array.ndim == 2:
                if mono_conversion_mode == 'average':
                    audio_array = np.mean(audio_array, axis=1)
                elif mono_conversion_mode == 'alternate':
                    audio_array = np.vstack((audio_array[:, 0], audio_array[:, 1])).reshape((-1,), order='F')
                else:
                    raise ValueError(f"Неверное значение параметра mono_conversion_mode: {mono_conversion_mode}")
                num_channels = 1
            elif audio_array.ndim == 1:
                num_channels = 1
            else:
                raise ValueError(f"Неподдерживаемая форма массива аудио: {audio_array.shape}")

            duration = len(audio_array) / sample_rate

            return {
                'audio_numpy': audio_array.astype(np.float32),
                'frequency': int(sample_rate),
                'num_channels': num_channels,
                'duration': duration
            }
        except Exception as e:
            logger.error(f"Ошибка при обработке аудиомассива: {e}")
            raise RuntimeError(f"Ошибка при обработке аудиомассива: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        """Коллбэк для sounddevice. Получает аудиофреймы и помещает их в очередь."""
        if status:
            logger.warning(f"Статус аудио: {status}")
        
        try:
            # Неблокирующее добавление в очередь
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # logger.warning("Очередь аудио переполнена, пропускаем фрейм")
            # Удаляем старый элемент и добавляем новый
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(indata.copy())
            except queue.Empty:
                pass

    def stt_worker(self):
        """Фоновый поток для обработки аудио из очереди."""
        logger.info("⏺️ Начался захват микрофона. Говорите...")
        buffer_size_samples = SAMPLE_RATE * BUFFER_DURATION
        
        while self.is_running:
            try:
                # Таймаут для возможности остановки потока
                data = self.audio_queue.get(timeout=1.0)
                data = data.flatten()

                self.buffer = np.concatenate((self.buffer, data))

                # Обработка каждые BUFFER_DURATION секунд аудио
                if len(self.buffer) >= buffer_size_samples:
                    logger.info(f"📤 Обработка фрагмента длиной {BUFFER_DURATION} секунд...")

                    try:
                        # Берем точно нужное количество сэмплов
                        audio_chunk = self.buffer[:buffer_size_samples]
                        
                        audio_data = self.read_audio_from_array(
                            audio_chunk, 
                            sample_rate=SAMPLE_RATE, 
                            mono_conversion_mode='average'
                        )
                        
                        result = self.stt_model.transcribe(
                            audio_array=audio_data['audio_numpy'], 
                            frequency=audio_data['frequency'])
                        
                        if result and result.get('text', '').strip():
                            logger.info(f"🗣️ Распознанный текст: {result['text']}")
                        else:
                            logger.info("🔇 Тишина или неразборчивая речь")
                            
                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки STT: {e}")

                    # Сдвигаем буфер (оставляем последние 1 секунду для плавности)
                    overlap_samples = SAMPLE_RATE  # 1 секунда
                    self.buffer = self.buffer[buffer_size_samples - overlap_samples:]
                    
            except queue.Empty:
                # Таймаут - продолжаем цикл
                continue
            except Exception as e:
                logger.error(f"❌ Неожиданная ошибка в worker: {e}")
                
        logger.info("🛑 STT worker остановлен")

    def start(self):
        """Запуск аудиопроцессора."""
        if self.is_running:
            logger.warning("Аудиопроцессор уже запущен")
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self.stt_worker, daemon=True)
        self.worker_thread.start()
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=DEVICE,
                channels=CHANNELS,
                dtype='float32',
                callback=self.audio_callback
            ):
                logger.info("🎤 Аудиопоток запущен. Нажмите Ctrl+C для остановки.")
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Получен сигнал остановки...")
        except Exception as e:
            logger.error(f"❌ Ошибка аудиопотока: {e}")
        finally:
            self.stop()

    def stop(self):
        """Остановка аудиопроцессора."""
        logger.info("🛑 Остановка аудиопроцессора...")
        self.is_running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            if self.worker_thread.is_alive():
                logger.warning("Worker поток не завершился в срок")

def main():
    """Главная функция."""
    try:
        # Проверка доступности устройств
        devices = sd.query_devices()
        logger.info("Доступные аудиоустройства:")
        for i, device in enumerate(devices):
            logger.info(f"  {i}: {device['name']} ({'вход' if device['max_input_channels'] > 0 else 'выход'})")
        
        # Создание и запуск процессора
        model_path = r'../models/faster-whisper-small'
        processor = AudioProcessor(model_path, compute_type='int8', device='cpu', target_frequency=SAMPLE_RATE)
        processor.start()
        
    except FileNotFoundError:
        logger.error("❌ Модель STT не найдена. Проверьте путь к модели.")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")

if __name__ == '__main__':
    main()