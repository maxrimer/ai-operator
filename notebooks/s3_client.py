from typing import Any, List, Optional
import os

from loguru import logger
import boto3

from s3_config import S3Config

class S3Client:
    def __init__(self, config: S3Config) -> None:
        """
        Инициализирует клиент S3 с заданной конфигурацией S3Config.
        Клиент может выполнять следующие операции:
            - Загрузка файла в бакет S3 (С локального хранилища)
            - Загрузка объекта в бакет S3
            - Скачивание файла из бакета S3
            - Удаление файла из бакета S3
            - Перечисление всех файлов в бакете S3
            - Проверка существования файла в бакете S3
            - Удаление всех файлов из бакета S3
            - Создание бакета S3
            - Удаление бакета S3
            

        Args:
            config: Объект конфигурации S3Config.
        """
        self.bucket_name = config.bucket_name
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            endpoint_url=config.endpoint_url
        )
        logger.info("S3 клиент успешно создан")

    def upload_file(self, local_file_path: str, s3_object_key: Optional[str] = None) -> None:
        """
        Загружает файл в указанный бакет S3.

        Args:
            local_file_path: Путь к локальному файлу для загрузки.
            s3_object_key: Ключ объекта в S3 (имя файла в бакете). 
                           Если None, используется имя локального файла.
        
        Raises:
            Exception: Исключение, вызванное ошибкой загрузки файла в S3 (botocore.exceptions).
        """
        if s3_object_key is None:
            s3_object_key = local_file_path
        try:
            self.s3.upload_file(local_file_path, self.bucket_name, s3_object_key)
            logger.info(f"Файл '{local_file_path}' успешно загружен в бакет '{self.bucket_name}' как '{s3_object_key}'")
        except Exception as e:
            e.add_note(f"Ошибка при загрузке файла '{local_file_path}' в S3")
            raise

    def download_file(self, s3_object_key: str, local_file_path: Optional[str] = None) -> None:
        """
        Скачивает файл из бакета S3.

        Args:
            s3_object_key: Ключ объекта в S3 (имя файла).
            local_file_path: Локальный путь для сохранения файла. Если None, используется имя из S3.
        
        Raises:
            Exception: Исключение, вызванное ошибкой скачивания файла из S3 (botocore.exceptions).
        """
        if local_file_path is None:
            local_file_path = s3_object_key
        try:
            self.s3.download_file(self.bucket_name, s3_object_key, local_file_path)
            logger.info(f"Файл '{s3_object_key}' успешно скачан из бакета '{self.bucket_name}' в '{local_file_path}'")
        except Exception as e:
            e.add_note(f"Ошибка при скачивании файла '{s3_object_key}' из S3")
            raise e

    def delete_file(self, s3_object_key: str) -> None:
        """
        Удаляет файл из бакета S3.

        Args:
            s3_object_key: Ключ объекта в S3 (имя файла для удаления).
        
        Raises:
            Exception: Исключение, вызванное ошибкой удаления файла из S3 (botocore.exceptions).
        """
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=s3_object_key)
            logger.info(f"Файл '{s3_object_key}' удален из бакета '{self.bucket_name}'")
        except Exception as e:
            e.add_note(f"Ошибка при удалении файла '{s3_object_key}' из S3")
            raise e
    
    def list_files(self) -> List[str]:
        """
        Перечисляет все файлы в указанном бакете S3.

        Returns:
            List[str]: Список ключей объектов (имен файлов) в бакете S3.
        
        Raises:
            Exception: Исключение, вызванное ошибкой получения списка файлов из S3 (botocore.exceptions).
        """
        try:
            object_keys = []
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' in page:
                    for obj in page["Contents"]:
                        object_keys.append(obj['Key'])
            logger.info(f"Получен список файлов из бакета '{self.bucket_name}'")
            return object_keys
        except Exception as e:
            e.add_note(f"Ошибка при получении списка файлов из S3 для бакета '{self.bucket_name}'")
            raise e

    def file_exists(self, s3_object_key: str) -> bool:
        """
        Проверяет, существует ли файл в бакете S3.

        Args:
            s3_object_key: Ключ объекта в S3 (имя файла для проверки).
        
        Returns:
            bool: True, если файл существует, иначе False.
        
        Raises:
            Exception: Исключение, вызванное ошибкой проверки наличия файла в S3 (botocore.exceptions).
        """
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_object_key)
            logger.info(f"Файл '{s3_object_key}' существует в бакете '{self.bucket_name}'")
            return True
        except Exception as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"Файл '{s3_object_key}' не найден в бакете '{self.bucket_name}'")
                return False
            else:
                e.add_note(f"Ошибка при проверке существования файла '{s3_object_key}' в S3")
                raise e
    
    def create_bucket(self, bucket_name: str) -> None:
        """
        Создает бакет в S3.

        Args:
            bucket_name: Имя бакета для создания.
        
        Raises:
            Exception: Исключение, вызванное ошибкой создания бакета S3 (botocore.exceptions).
        """
        try:
            self.s3.create_bucket(Bucket=bucket_name)
            logger.info(f"Бакет '{bucket_name}' успешно создан")
        except Exception as e:
            e.add_note(f"Ошибка при создании бакета '{bucket_name}'")
            raise e
    
    def delete_objects(self) -> None:
        """
        Удаляет все объекты из бакета S3.
        
        Raises:
            Exception: Исключение, вызванное ошибкой удаления объектов из S3 (botocore.exceptions).
        """
        objects = self.s3.get_paginator("list_objects_v2")
        objects_iterator = objects.paginate(Bucket=self.bucket_name)

        try:
            for page in objects_iterator:
                if "Contents" in page:
                    objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
                    self.s3.delete_objects(Bucket=self.bucket_name, Delete={"Objects": objects_to_delete})
            logger.info(f"Все объекты из бакета '{self.bucket_name}' успешно удалены")
        except Exception as e:
            e.add_note(f"Ошибка при удалении объектов из S3 для бакета '{self.bucket_name}'")
            raise e

    def delete_bucket(self, delete_if_not_empty: bool = False) -> None:
        """
        Удаляет бакет S3.

        Args:
            delete_if_not_empty: Если True, удаляет все объекты в бакете перед удалением бакета.
        
        Raises:
            Exception: Исключение, вызванное ошибкой удаления бакета S3 (botocore.exceptions).
        """
        if delete_if_not_empty:
            self.delete_objects()

        try:
            self.s3.delete_bucket(Bucket=self.bucket_name)
            logger.info(f"Бакет '{self.bucket_name}' успешно удален")
        except Exception as e:
            e.add_note(f"Ошибка при удалении бакета '{self.bucket_name}'")
            raise e

    def upload_object(self, object_key: str, object_data: bytes) -> None:
        """
        Загружает объект в бакет S3 с указанным ключом и данными.

        Args:
            object_key: Ключ объекта в S3 (имя файла).
            object_data: Данные объекта в байтах.
        
        Raises:
            Exception: Исключение, вызванное ошибкой загрузки объекта в S3 (botocore.exceptions).
        """
        try:
            self.s3.put_object(Body=object_data, Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Объект с ключом '{object_key }' успешно загружен в бакет '{self.bucket_name}'")
        except Exception as e:
            e.add_note(f"Ошибка при загрузке объекта '{object_key}' в S3")
            raise e