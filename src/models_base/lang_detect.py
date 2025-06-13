import functools
import pathlib

import fasttext


_MODEL_PATH = pathlib.Path(__file__).with_name("lid.176.bin")


@functools.lru_cache(maxsize=1)
def get_lang_model():
    return fasttext.load_model(str(_MODEL_PATH))


def detect_lang(text: str) -> str:
    mdl = get_lang_model()
    return mdl.predict(text.replace("\n", " "))[0][0].split("__")[-1]


