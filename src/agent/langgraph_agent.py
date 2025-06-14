import json
import ast
import re
from typing import Annotated, List

from src import hint_validator_node, search_kb, similar_case, acc_info_retriever_tool, acc_blocks_retriever_tool, \
                retrieve_account_info, retrieve_bloks_info
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from loguru import logger
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv


from src.agent.llm_wrapper import call_external_llm
from src.agent.prompts import generate_query_for_kb, generate_clarify_validation_prompt, generate_clarification_prompt, \
                    generate_final_response

load_dotenv()

from src.retriever.csv_retriever import ALIAS_PATH
with open(ALIAS_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
values_list = list(data.values())

model = call_external_llm(model_name="gpt-4o")

tools = [search_kb, similar_case, acc_info_retriever_tool, acc_blocks_retriever_tool]

model_with_tools = model.bind_tools(tools)

memory = MemorySaver()


class CallState(BaseModel):
    customer_query: str
    customer_id: int
    is_query_need_clarification: bool = False
    query_for_kb: str = ""
    hint: str | None = None
    source: str | None = None
    confidence: float = 0.0
    hint_valid: bool = True
    validator_msg: str = ""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)


sg = StateGraph(CallState)


def detect_clarification(state: CallState) -> CallState:
    logger.info(f'Started #1 State: detect_clarification')
    state.messages.append(HumanMessage(content=state.customer_query))
    prompt = generate_clarify_validation_prompt(values_list)
    messages = [SystemMessage(content=prompt)] + state.messages
    resp = model.invoke(messages)
    state.is_query_need_clarification = resp.content.lower().startswith('да')
    logger.info(f'Finished #1 State: {state.is_query_need_clarification}; {resp}')
    return state


def rewrite_query(state: CallState) -> CallState:
    logger.info(f'Started #2 State: rewrite_query')
    prompt = generate_query_for_kb()
    messages = [SystemMessage(content=prompt)] + state.messages
    text = model.invoke(messages)
    clean = text.content.strip().strip('"')
    state.query_for_kb = clean
    logger.info(f'Finished #2 State: {text.content}')
    return state


def needs_clarification(state: CallState) -> str:
    if state.is_query_need_clarification:
        return 'AskClarificationPlease'
    else:
        return 'NoClarificationNeeded'


def ask_clarification(state: CallState) -> CallState:
    logger.info(f'Started #2 State: ask_clarification')
    prompt = generate_clarification_prompt(state)
    messages = [SystemMessage(content=prompt)] + state.messages
    resp = model.invoke(messages)
    state.messages.append(resp)
    state.hint = resp.content.strip()
    logger.info(f'Finished #2 State: {resp}')
    return state


def generate_hint(state: CallState) -> CallState:
    logger.info(f'Started #3 State: generate_hint')
    prompt = generate_final_response(state, values_list)
    messages = [SystemMessage(content=prompt)] + state.messages

    resp1 = model_with_tools.invoke(messages, tool_choice="auto")
    logger.info(f'Started #3 State: invoked model_with_tools')
    tool_calls = resp1.additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError("LLM не запросил инструменты")

    tool_outputs = []
    for tc in tool_calls:
        fn, args = tc["function"]["name"], json.loads(tc["function"]["arguments"])
        if fn == "search_kb":
            logger.info(f'start search_kb')
            result = search_kb(**args)
            logger.info(f'search_kb: {result}')
        elif fn == "similar_case":
            logger.info(f'start similar_case')
            result = similar_case(**args)
            logger.info(f'similar_case: {result}')
        elif fn == "retrieve_account_info":
            if "__arg1" in args and "client_id" not in args:
                args["client_id"] = args.pop("__arg1")
            df = retrieve_account_info(args["client_id"])
            result = df.to_dict(orient="records")
        elif fn == "retrieve_bloks_info":
            if "__arg1" in args and "client_id" not in args:
                args["client_id"] = args.pop("__arg1")
            df = retrieve_account_info(args["client_id"])
            result = df.to_dict(orient="records")
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
    logger.info(f'invoke resp2')
    resp2 = model_with_tools.invoke(messages, tool_choice="none")
    content = resp2.content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```[a-z]*\s*", "", content, flags=re.I)
        content = re.sub(r"\s*```$", "", content).strip()
    if not content.startswith("{"):
        content = "{" + content
    if not content.endswith("}"):
        content = content + "}"
    try:
        final = json.loads(content)
        state.hint = final.get("hint")
        state.confidence = final.get("confidence")
        state.source = final.get("source")
        agent_reply = AIMessage(content=final.get("hint"))
        state.messages.append(agent_reply)
        logger.info(f'Finished #3 State: {final}')
    except json.JSONDecodeError:
        try:
            final = ast.literal_eval(content)
        except Exception:
            logger.error(f"Bad LLM output, cannot parse JSON:\n{content}")

    return state


sg.add_node("DetectClarification", detect_clarification)
sg.add_node("RewriteQuery", rewrite_query)
sg.add_node("AskClarification", ask_clarification)
sg.add_node("GenerateHint", generate_hint)
# sg.add_node("HintValidator", hint_validator_node)

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
sg.add_edge("GenerateHint", END)


flow = sg.compile(checkpointer=memory)

config = {'configurable': {'thread_id': '126'}}


if __name__ == "__main__":
    for utt in [
        """
        Я только что онлайн оформила через приложение кредит. И все нормально прошло.
        И написано, что деньги уже поступили. А куда поступили, я не могу найти.
        На текущий счет?  На текущем счету нет.
        """
    ]:
        st = CallState(customer_query=utt, customer_id=77017563318)
        result = flow.invoke(st, config)
        print(result)





    # init_state = CallState(customer_query=customer_query, customer_id=customer_id)
    # result = flow.invoke(init_state, config)
    # print(result)


















