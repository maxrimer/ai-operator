from llm_wrapper import call_local_llm
from prompts import generate_query_for_kb, generate_clarify_prompt, generate_query_valid_prompt
from src import hint_validator_node
from src import is_potential_query

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
import numpy as np


KW_NOUNS = ["кредит", "карта", "комиссия", "погашение", "лимит", "перевод", "ипотека"]


class CallState(BaseModel):
    partial: str
    buffer: str = ""
    is_query: bool = True
    history: list[str] = []
    utt: str = ""
    query: str = ""
    best_doc: str | None = None
    sim: float = 0.0
    hint: str | None = None
    llm_lp: float | None = None
    hint_quality: float = 0.0
    kw_len: int = 0
    unk_ratio: float = 0.0
    hint_valid: bool = True
    validator_msg: str = ""


sg = StateGraph(CallState)


def detect_question(state: CallState) -> CallState:
    candidate = f"{state.buffer} {state.partial}".strip()
    print(candidate)
    is_q = is_potential_query(text=candidate, kw_nouns=KW_NOUNS)
    if not is_q:
        prompt = generate_query_valid_prompt(candidate)
        ans = call_local_llm(prompt, max_tokens=1, temperature=0.0)
        is_q = ans.strip().lower().startswith("да")
    state.is_query = is_q
    return state


def should_accumulate_or_run(state: CallState) -> str:
    is_q = state.is_query

    if is_q:
        return "continue"
    else:
        return "accumulate"


def accumulate_buffer(state: CallState) -> CallState:
    """
    Сохраняем текущий кусок в buffer и ждём новых сегментов.
    """
    state.buffer = f"{state.buffer} {state.partial}".strip()
    state.partial = ""
    return state


def rewrite_query(state: CallState) -> CallState:
    state.utt = f"{state.buffer} {state.partial}".strip()
    state.buffer = ""
    state.partial = ""
    prompt = generate_query_for_kb(state)
    text = call_local_llm(prompt, max_tokens=100, temperature=0.0)
    clean = text.strip().strip('"').replace("Ключевые слова:", "")
    state.query = clean
    state.kw_len = len(clean.split())
    state.unk_ratio = clean.count("unknown") / max(1, state.kw_len)
    state.history.append(state.utt)
    return state


def needs_clarification(state: CallState) -> bool:
    return (
        state.kw_len < 3
        or state.unk_ratio > 0.35
    )


def ask_clarification(state: CallState) -> CallState:
    prompt = generate_clarify_prompt(state)
    text = call_local_llm(prompt, max_tokens=120, temperature=0.0)
    state.hint = text.strip()
    state.hint_quality = 0.3
    return state


def retrieve_doc(state: CallState) -> CallState:
    q = state.query or state.utt
    # emb = model.encode([q], normalize_embeddings=True)
    # D, I = index.search(np.array(emb), k=1)
    state.best_doc = None
    state.sim = 0.0
    return state


def generate_hint(state: CallState) -> CallState:
    state.hint = f"MOCK_ANSWER({state.utt[:30]})"
    state.llm_lp = -0.7
    return state


def score_quality(state: CallState) -> CallState:
    score_asr = 0.7
    score_ret = state.sim / 0.4
    score_llm = np.exp(state.llm_lp)
    state.hint_quality = 0.4*score_asr + 0.3*score_ret + 0.3*score_llm
    return state


sg.add_node("DetectQuestion", detect_question)
sg.add_node("RewriteQuery", rewrite_query)
sg.add_node("AccumulateBuffer", accumulate_buffer)
sg.add_node("AskClarification", ask_clarification)
sg.add_node("RetrieveDoc", retrieve_doc)
sg.add_node("GenerateHint", generate_hint)
sg.add_node("ScoreQuality",  score_quality)
sg.add_node("HintValidator", hint_validator_node)


sg.add_edge(START, "DetectQuestion")
sg.add_conditional_edges(
    "DetectQuestion",
    should_accumulate_or_run,
    {
        "continue": "RewriteQuery",
        "accumulate": "AccumulateBuffer"
    }
)
sg.add_edge("AccumulateBuffer", END)
sg.add_conditional_edges(
    "RewriteQuery",
    needs_clarification,
    {
        True: "AskClarification",
        False: "RetrieveDoc"
    }
)
sg.add_edge("AskClarification",  END)
sg.add_edge("RetrieveDoc", "GenerateHint")
sg.add_edge("GenerateHint", "ScoreQuality")
sg.add_edge("ScoreQuality", "HintValidator")
sg.add_edge("HintValidator",  END)


flow = sg.compile()


if __name__ == "__main__":
    for seg in ["Здравствуйте, я хотел бы узнать",
                "сколько осталось платить по кредиту?"]:
        init_state = CallState(partial=seg)
        diff = flow.invoke(init_state)

        init_state = init_state.model_copy(update=diff)

        print(init_state)









    # init_state = CallState(partial='Здравствуйте, я хотел бы узнать')
    # result = flow.invoke(init_state)
    # print(result)
    #
    # print("*" * 40)
    #
    # init_state.partial = "сколько осталось платить по кредиту?"
    # result2 = flow.invoke(init_state)
    # # теперь граф должен был выдать hint и очистить buffer
    # print(result2)
















