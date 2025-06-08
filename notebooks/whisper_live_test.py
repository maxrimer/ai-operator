from whisper_live.client import TranscriptionClient

# Создание клиента
client = TranscriptionClient(
    "localhost",
    9090,
    lang="ru",
    translate=False,
    model="small",
    use_vad=False,
    save_output_recording=True,
    output_recording_filename="./output_recording.wav",
    max_clients=4,
    max_connection_time=600,
    mute_audio_playback=False,
)

# Запуск транскрипции с микрофона
print("Начинаем транскрипцию с микрофона...")
print("Нажмите Ctrl+C для остановки")

try:
    # Для транскрипции с микрофона
    client()
    
    # Альтернативно, для транскрипции файла:
    # client("path/to/your/audio/file.wav")
    
except KeyboardInterrupt:
    print("\nТранскрипция остановлена")
except Exception as e:
    print(f"Ошибка: {e}")