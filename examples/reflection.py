import os
import getpass
from typing import List
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages


def _set_if_undefined(var: str):
    """Prompts for environment variables if they are not set."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please enter your {var}: ")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGFUSE_PUBLIC_KEY")
_set_if_undefined("LANGFUSE_SECRET_KEY")


llm = ChatOpenAI(model="deepseek-chat", temperature=0)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            " Provide detailed recommendations, including requests for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm


class State(TypedDict):
    # The state tracks the list of messages exchanged
    messages: Annotated[List[BaseMessage], add_messages]


def generation_node(state: State):
    """
    Node that generates the essay based on the current conversation history.
    """
    return {"messages": [generate_chain.invoke(state["messages"])]}


def reflection_node(state: State):
    """
    Node that reflects on the most recent message (the essay) and generates a critique.
    It maps the last AI message to a HumanMessage so the 'teacher' LLM sees it as a submission.
    """
    cls_map = {"ai": HumanMessage, "human": AIMessage}

    last_message = state["messages"][-1]
    translated_message = cls_map[last_message.type](content=last_message.content)

    critique_input = [state["messages"][0], translated_message]

    res = reflect_chain.invoke(critique_input)

    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: State):
    """
    Condition to check if we should loop back or stop.
    Stops after 6 messages (approx 3 cycles of generate -> reflect).
    """
    if len(state["messages"]) > 2:
        return END
    return "reflect"


builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)

builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")


graph = builder.compile()


def main():
    user_request = (
        "Write an essay on why the little prince is relevant in modern childhood"
    )
    inputs = {"messages": [HumanMessage(content=user_request)]}

    graph.stream(inputs)

    graph.invoke(inputs)


if __name__ == "__main__":
    main()
