import os
import pandas as pd
from langchain.tools import Tool
import json
from pathlib import Path
import tempfile
from src.utils.s3_config import S3Config
from src.utils.s3_client import S3Client

config = S3Config(
    endpoint_url=os.environ['S3_URL'],
    access_key_id=os.environ['S3_AK'],
    secret_access_key=os.environ['S3_SK'],
    bucket_name='data-for-case-1'
)
s3_client = S3Client(config=config)


data_path_krb = 'AI-суфлер общий доступ/КРБ/Данные/resultfizFinal Final.csv'
data_path_mmb = 'AI-суфлер общий доступ/ММБ/Данные/DBZURRESULTFinal.csv'
data_path_dop_mmb = 'AI-суфлер общий доступ/ММБ/Данные/FINALresultURAcctsAndBLocksFinal.csv'
ALIAS_PATH = Path(__file__).parent.parent / 'configs' / 'aliases.json'

_ALIASES     = None


def _aliases():
    global _ALIASES
    if _ALIASES is None:
        _ALIASES = json.loads(ALIAS_PATH.read_text(encoding="utf-8"))
    return _ALIASES


def _apply_aliases(row: pd.Series) -> dict:
    rename = _aliases()
    out = {}
    for k, v in row.items():
        key = rename.get(k, k)
        out[key] = v
    return out


def load_account_csv():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, dir='/tmp') as tmp_krb, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False, dir='/tmp') as tmp_mmb:
        local_file_path_krb = tmp_krb.name
        local_file_path_mmb = tmp_mmb.name
        
        s3_client.download_file(data_path_krb, local_file_path_krb)
        s3_client.download_file(data_path_mmb, local_file_path_mmb)
        
        krb = pd.read_csv(local_file_path_krb, encoding='cp1251', sep=';', engine='python')
        mmb = pd.read_csv(local_file_path_mmb, encoding='cp1251', sep=';', engine='python')
        
        # Clean up temporary files
        os.unlink(local_file_path_krb)
        os.unlink(local_file_path_mmb)

    mmb.rename(columns={'CALL_ID': 'ID'}, inplace=True)
    krb.rename(columns={'Номер телефона': 'ID'}, inplace=True)
    all_data = pd.concat([krb, mmb])
    return all_data


def load_bloks_csv():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, dir='/tmp') as tmp_mmb:
        local_file_path__dop_mmb = tmp_mmb.name
        s3_client.download_file(data_path_mmb, local_file_path__dop_mmb)

        mmd_dop = pd.read_csv(local_file_path__dop_mmb, encoding='cp1251',
                            sep=';', engine='python')
        
        # Clean up temporary file
        os.unlink(local_file_path__dop_mmb)
        
        mmd_dop.drop_duplicates(inplace=True)
        return mmd_dop


def retrieve_account_info(client_id: int) -> dict:
    all_data = load_account_csv()
    rows = all_data[all_data.ID == client_id]
    if rows.empty:
        return {"client_id": client_id, "accounts": []}

    accounts = [_apply_aliases(r) for _, r in rows.iterrows()]
    return {"client_id": client_id, "accounts": accounts}


def retrieve_bloks_info(client_id: int) -> dict:
    bloks_data = load_bloks_csv()
    rows = bloks_data[bloks_data.call == client_id]
    if rows.empty:
        return {"client_id": client_id, "accounts": []}
    keep_cols = [c for c in bloks_data.columns
                 if c != "call"]
    accounts = rows[keep_cols].to_dict(orient="records")
    return {"client_id": client_id, "accounts": accounts}


acc_info_retriever_tool = Tool(
    name="retrieve_account_info",
    func=retrieve_account_info,
    description="Возвращает информацию о состоянии аккаунта/арестах/просрочках/"
                "задолжностях/открытых продуктах и прочим по id клиента"
)


acc_blocks_retriever_tool = Tool(
    name="retrieve_bloks_info",
    func=retrieve_bloks_info,
    description="Возвращает информацию о 3 аттрибутах: Номер счета,"
                "Остаток на счете, Типы блокировок по id клиента"
)


if __name__ == '__main__':
    info = retrieve_bloks_info(74957)
    print(info['accounts'])


