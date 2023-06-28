from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import discord

from langchain.agents import AgentType, initialize_agent, OpenAIFunctionsAgent, Tool
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import YouTubeSearchTool
from langchain.utilities import GoogleSearchAPIWrapper

from src.base import Message
from src.constants import BOT_NAME, SYSTEM_MESSAGE
from src.moderation import (
    moderate_message,
    send_moderation_flagged_message,
    send_moderation_blocked_message,
)
from src.utils import split_into_shorter_messages, close_thread, logger

gsearch = GoogleSearchAPIWrapper()
ytsearch = YouTubeSearchTool()
tools = [
    Tool(
        name = "google-search",
        func=gsearch.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name = "YouTube-search",
        func=ytsearch.run,
        description="useful for when you need to look for video clips"
    )
]
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
system_message = SystemMessage(
    content=SYSTEM_MESSAGE
)
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limt=500, memory_key="memory", return_messages=True)

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


async def generate_completion_response(
    messages: List[Message], user: str
) -> CompletionData:
    try:
        rendered = messages[-1].render()
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory, verbose=True)
        reply = agent.run(input=rendered)
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
