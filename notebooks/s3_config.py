import os
from typing import Optional


class S3Config:
    """
        Конфигурация для подключения к AWS S3.

        Этот класс предоставляет конфигурацию для подключения к AWS S3, включая URL конечной точки, идентификатор ключа доступа,
        секретный ключ доступа и имя бакета. Конфигурация может быть инициализирована явно через параметры конструктора
        или через переменную среды, которая содержит все необходимые значения, разделенные ';;'.

        Атрибуты:
            endpoint_url (Optional[str]): URL конечной точки S3.
            access_key_id (Optional[str]): Идентификатор ключа доступа к S3.
            secret_access_key (Optional[str]): Секретный ключ доступа к S3.
            bucket_name (Optional[str]): Имя бакета S3.
        """
    def __init__(
            self,
            environment_val: Optional[str] = None,
            endpoint_url: Optional[str] = None,
            access_key_id: Optional[str] = None,
            secret_access_key: Optional[str] = None,
            bucket_name: Optional[str] = None,
    ):
        self.endpoint_url, self.access_key_id, self.secret_access_key, self.bucket_name = endpoint_url, access_key_id, secret_access_key, bucket_name

        if not (environment_val is None):
            params = os.environ[environment_val]
            self.endpoint_url, self.access_key_id, self.secret_access_key, self.bucket_name = params.split(';;')