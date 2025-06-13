from urllib.parse import quote
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.configs import DbConfig

conf = DbConfig()

# Подключение к базе
SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{conf.DB_USER}:{quote(conf.DB_PASSWORD)}@{conf.DB_HOST}:{conf.DB_PORT}/{conf.DB_NAME}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    pool_size=10,         # Максимум 10 соединений в пуле
    max_overflow=5,       # Дополнительно можно создать еще 5 соединений
    pool_timeout=20,      # Максимум 20 секунд ожидания соединения
    pool_recycle=1800,    # Перезапуск соединений каждые 30 минут
    pool_pre_ping=True    # Проверка соединений перед использованием
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """
    Настраивает менеджер контекста, используя yield ключевое слово.
    Он создает сеанс базы данных (db) с помощью SessionLocal и передает его вызывающей стороне.
    После завершения выполнения в контексте finally блок обеспечивает закрытие сеанса.
    """

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
