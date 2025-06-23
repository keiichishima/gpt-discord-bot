import asyncio
from dataclasses import dataclass
from enum import Enum
import json
from typing import Optional, List

import discord

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI

from src.base import Message
from src.constants import OPENAI_MODEL, BOT_NAME, SYSTEM_MESSAGE, HUMAN_MESSAGE
from src.moderation import (
    moderate_message,
    send_moderation_flagged_message,
    send_moderation_blocked_message,
)
from src.utils import split_into_shorter_messages, close_thread, logger

model = ChatOpenAI(temperature=0, model=OPENAI_MODEL)

gsearch = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "google-search",
        func = gsearch.run,
        description = "useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    YouTubeSearchTool(),
    OpenWeatherMapQueryRun(api_wrapper=OpenWeatherMapAPIWrapper())
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    MessagesPlaceholder(variable_name="chat_history", optional=True),  # 会話履歴を挿入
    ("human", HUMAN_MESSAGE)  # 最新のユーザー入力
])
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    handle_parsing_errors=True,
    verbose=True
)

class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3
    MODERATION_FLAGGED = 4
    MODERATION_BLOCKED = 5


@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]

def render_messages(messages: List[Message]) -> List[BaseMessage]:
    rendered = []
    for m in messages:
        if m.user.startswith(BOT_NAME):
            rendered.append(AIMessage(content=f"{m.user}: {m.text}"))
        else:
            rendered.append(HumanMessage(content=f"{m.user}: {m.text}"))
    return rendered

async def generate_completion_response(
    messages: List[Message], user: str
) -> CompletionData:
    try:
        loop = asyncio.get_running_loop()
        rendered = messages[-1].render()
        reponse = await loop.run_in_executor(None, lambda: agent_executor.invoke(
            {
                "input": rendered,
                "chat_history": render_messages(messages),
            }
        ))
        reply = reponse["output"]
        if reply:
            flagged_str, blocked_str = moderate_message(
                message=(rendered + reply)[-500:], user=user
            )
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )

            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )

        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )
    except Exception as e:
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(e)
        )


async def process_response(
    user: str, thread: discord.Thread, response_data: CompletionData
):
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text
    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        sent_message = None
        if not reply_text:
            sent_message = await thread.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        else:
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                sent_message = await thread.send(r)
        if status is CompletionResult.MODERATION_FLAGGED:
            await send_moderation_flagged_message(
                guild=thread.guild,
                user=user,
                flagged_str=status_text,
                message=reply_text,
                url=sent_message.jump_url if sent_message else "no url",
            )

            await thread.send(
                embed=discord.Embed(
                    description=f"⚠️ **This conversation has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )
    elif status is CompletionResult.MODERATION_BLOCKED:
        await send_moderation_blocked_message(
            guild=thread.guild,
            user=user,
            blocked_str=status_text,
            message=reply_text,
        )

        await thread.send(
            embed=discord.Embed(
                description=f"❌ **The response has been blocked by moderation.**",
                color=discord.Color.red(),
            )
        )
    elif status is CompletionResult.TOO_LONG:
        await close_thread(thread)
    elif status is CompletionResult.INVALID_REQUEST:
        await thread.send(
            embed=discord.Embed(
                description=f"**Invalid request** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
    else:
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
