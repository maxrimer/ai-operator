from typing import Dict, Any

import librosa
import numpy as np
from faster_whisper import WhisperModel


class STTFasterWhisperModel:
    """
    Класс для распознавания речи с использованием ускоренной модели Whisper.

    Этот класс наследует от STTBaseModel и предназначен для работы с ускоренной версией
    модели Whisper, поддерживающей вычисления на CPU и GPU.

    Метрики: RU_WER ~ 5.0, KK_WER ~ 32.4
    """

    def __init__(self, 
                 path_to_model: str, 
                 compute_type, 
                 device: str = 'auto', 
                 target_frequency: int = 16_000) -> None:

        self.target_frequency = target_frequency
        print(f"device: {device}; Начало инициализации модели...")
        self.model = WhisperModel("small", device=device, compute_type=compute_type)
        print(f"device: {device}; Модель инициализирована")
        # self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    def _change_frequency(self, audio_array, current_sampling_rate, target_sampling_rate):
        resampled_audio_array = librosa.resample(audio_array, orig_sr=current_sampling_rate, target_sr=target_sampling_rate)
        return resampled_audio_array

    def transcribe(self, audio_array: np.ndarray, frequency) -> Dict[str, Any]:
        """
        Транскрибирует аудио-массив и возвращает словарь с текстом распознанной речи.

        Args:
            audio_array Массив аудио-сигнала.
            audio_metadata: Метаданные аудио, содержащие, например, частоту дискретизации.
            transcribe_settings: Настройки для транскрипции.

        Returns:
            Словарь с ключом 'text', содержащий текст распознанной речи.
        """

        resampled_audio_array = self._change_frequency(audio_array, frequency, self.target_frequency)

        segments, info = self.model.transcribe(resampled_audio_array)

        result_list = []
        for segment in segments:
            result_list.append(segment.text)

        transcription = ''.join(result_list)

        result_dict = {'text': transcription}

        return result_dict
