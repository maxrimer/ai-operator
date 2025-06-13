import json
from prompts import generate_query_for_kb, generate_clarify_validation_prompt, generate_clarification_prompt, \
                    generate_final_response
from src import hint_validator_node, search_kb, similar_case

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from dotenv import load_dotenv

from src.agent.llm_wrapper import call_external_llm

load_dotenv()

KW_NOUNS = ["кредит", "карта", "комиссия", "погашение", "лимит", "перевод", "ипотека"]

model = call_external_llm(model_name="gpt-4o")

tools = [search_kb, similar_case]

model_with_tools = model.bind_tools(tools)


class CallState(BaseModel):
    customer_query: str
    dialog_lang: str
    is_query_need_clarification: bool = False
    query_for_kb: str = ""
    hint: str | None = None
    source: str | None = None
    confidence: float = 0.0
    hint_valid: bool = True
    validator_msg: str = ""


sg = StateGraph(CallState)


def detect_clarification(state: CallState) -> CallState:
    prompt = generate_clarify_validation_prompt()
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state.customer_query)
    ]
    text = model.invoke(messages)
    state.is_query_need_clarification = text.content.lower().startswith('да')
    return state


def rewrite_query(state: CallState) -> CallState:
    prompt = generate_query_for_kb()
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state.customer_query)
    ]
    text = model.invoke(messages)
    clean = text.content.strip().strip('"')
    state.query_for_kb = clean
    return state


def needs_clarification(state: CallState) -> str:
    if state.is_query_need_clarification:
        return 'AskClarificationPlease'
    else:
        return 'NoClarificationNeeded'


def ask_clarification(state: CallState) -> CallState:
    prompt = generate_clarification_prompt(state)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state.customer_query)
    ]
    text = model.invoke(messages)
    state.hint = text.content.strip()
    return state


def generate_hint(state: CallState) -> CallState:
    prompt = generate_final_response(state)
    messages = [SystemMessage(content=prompt)]

    resp1 = model_with_tools.invoke(messages, tool_choice="auto")
    tool_calls = resp1.additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError("LLM не запросил инструменты")

    tool_outputs = []
    for tc in tool_calls:
        fn, args = tc["function"]["name"], json.loads(tc["function"]["arguments"])
        if fn == "search_kb":
            result = search_kb(**args)
        elif fn == "similar_case":
            result = similar_case(**args)
        else:
            result = {}

        tool_outputs.append(
            ToolMessage(
                tool_call_id=tc["id"],
                name=fn,
                content=json.dumps(result, ensure_ascii=False)
            )
        )

    messages += [resp1] + tool_outputs
    resp2 = model_with_tools.invoke(messages, tool_choice="none")

    try:
        final = json.loads(resp2.content)
        state.hint = final["hint"]
        state.confidence = final["confidence"]
        state.source = final["source"]
    except Exception as e:
        raise ValueError(f"Bad LLM output: {resp2.content}") from e

    return state


sg.add_node("DetectClarification", detect_clarification)
sg.add_node("RewriteQuery", rewrite_query)
sg.add_node("AskClarification", ask_clarification)
sg.add_node("GenerateHint", generate_hint)
sg.add_node("HintValidator", hint_validator_node)

sg.add_edge(START, "DetectClarification")
sg.add_conditional_edges(
    "DetectClarification",
    needs_clarification,
    {
        "AskClarificationPlease": "AskClarification",
        "NoClarificationNeeded": "RewriteQuery"
    }
)
sg.add_edge("AskClarification", END)
sg.add_edge("RewriteQuery", "GenerateHint")
sg.add_edge("GenerateHint", "HintValidator")
sg.add_edge("HintValidator",  END)


flow = sg.compile()


if __name__ == "__main__":
    customer_query = "Сәлем! Менде әлі де Сбербанктен Visa картасы бар, оның жарамдылық мерзімі аяқталмаған," \
                     "картам халықаралық төлемдерге ашық. Мен оны Қазақстанда әлі де қолдана аламын ба?"
    lang = 'kz'
    init_state = CallState(customer_query=customer_query, dialog_lang=lang)
    result = flow.invoke(init_state)
    print(result)

















