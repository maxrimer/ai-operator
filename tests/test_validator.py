from src import hint_validator_node
from src import CallState


def test_validator_lang():
    st = CallState(hint="Привет меня зовут Максим and love!", llm_lp=-0.3)
    out = hint_validator_node(st)
    print(out.hint_valid)
    print(out.validator_msg)


if __name__ == "__main__":
    test_validator_lang()

