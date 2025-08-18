# Discord bot completion handling module
# Manages AI agent responses and moderation for Discord threads

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

# OpenAI model instance with zero temperature for consistent responses
model = ChatOpenAI(temperature=0.1, model=OPENAI_MODEL)

# Google search wrapper for web search functionality
gsearch = GoogleSearchAPIWrapper()

# Available tools for the AI agent
tools = [
    Tool(
        name = "google-search",
        func = gsearch.run,
        description = "useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    YouTubeSearchTool(),  # For searching YouTube videos
    OpenWeatherMapQueryRun(api_wrapper=OpenWeatherMapAPIWrapper())  # For weather information
]

# Chat prompt template with system message and conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    MessagesPlaceholder(variable_name="chat_history", optional=True),  # Insert conversation history
    ("human", HUMAN_MESSAGE)  # Latest user input
])

# Create structured chat agent with LLM, tools, and prompt
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

# Agent executor with configuration for handling interactions
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # Maximum number of iterations per request
    handle_parsing_errors=True,  # Handle parsing errors gracefully
    verbose=True  # Enable verbose logging
)

class CompletionResult(Enum):
    """Enum representing different completion result statuses"""
    OK = 0                    # Successful completion
    TOO_LONG = 1             # Response too long
    INVALID_REQUEST = 2      # Invalid request format
    OTHER_ERROR = 3          # General error
    MODERATION_FLAGGED = 4   # Content flagged by moderation
    MODERATION_BLOCKED = 5   # Content blocked by moderation


@dataclass
class CompletionData:
    """Data class for completion response with status and content"""
    status: CompletionResult          # Result status
    reply_text: Optional[str]         # Generated reply text
    status_text: Optional[str]        # Additional status information

def render_messages(messages: List[Message]) -> List[BaseMessage]:
    """Convert Message objects to BaseMessage objects for LangChain
    
    Args:
        messages: List of Message objects from conversation history
        
    Returns:
        List of BaseMessage objects (AIMessage or HumanMessage)
    """
    rendered = []
    for m in messages:
        # Determine message type based on user name
        if m.user.startswith(BOT_NAME):
            rendered.append(AIMessage(content=f"{m.user}: {m.text}"))
        else:
            rendered.append(HumanMessage(content=f"{m.user}: {m.text}"))
    return rendered

async def generate_completion_response(
    messages: List[Message], user: str
) -> CompletionData:
    """Generate AI completion response using the agent executor
    
    Args:
        messages: List of conversation messages
        user: Username for moderation purposes
        
    Returns:
        CompletionData with status and response content
    """
    try:
        # Get current event loop for async execution
        loop = asyncio.get_running_loop()
        
        # Render the latest message for input
        rendered = messages[-1].render()
        
        # Execute agent with conversation history in thread pool
        response = await loop.run_in_executor(None, lambda: agent_executor.invoke(
            {
                "input": rendered,
                "chat_history": render_messages(messages[:-1]),  # All messages except latest
            }
        ))
        
        reply = response["output"]
        
        # Moderate the response if it exists
        if reply:
            # Check last 500 characters for moderation
            flagged_str, blocked_str = moderate_message(
                message=(rendered + reply)[-500:], user=user
            )
            
            # Handle blocked content
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )

            # Handle flagged content
            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )

        # Return successful response
        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )
    except Exception as e:
        # Log exception and return error status
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(e)
        )


async def process_response(
    user: str, thread: discord.Thread, response_data: CompletionData
):
    """Process and send the completion response to Discord thread
    
    Args:
        user: Username for moderation reporting
        thread: Discord thread to send response to
        response_data: CompletionData containing status and response
    """
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text
    
    # Handle successful or flagged responses
    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        sent_message = None
        
        # Handle empty response
        if not reply_text:
            sent_message = await thread.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        else:
            # Split long messages and send them
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                sent_message = await thread.send(r)
        
        # Handle moderation flagged content
        if status is CompletionResult.MODERATION_FLAGGED:
            # Send moderation notification to appropriate channel
            await send_moderation_flagged_message(
                guild=thread.guild,
                user=user,
                flagged_str=status_text,
                message=reply_text,
                url=sent_message.jump_url if sent_message else "no url",
            )

            # Notify user about flagged content
            await thread.send(
                embed=discord.Embed(
                    description=f"⚠️ **This conversation has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )
    
    # Handle blocked content
    elif status is CompletionResult.MODERATION_BLOCKED:
        # Send moderation blocked notification
        await send_moderation_blocked_message(
            guild=thread.guild,
            user=user,
            blocked_str=status_text,
            message=reply_text,
        )

        # Notify user about blocked content
        await thread.send(
            embed=discord.Embed(
                description=f"❌ **The response has been blocked by moderation.**",
                color=discord.Color.red(),
            )
        )
    
    # Handle response too long
    elif status is CompletionResult.TOO_LONG:
        await close_thread(thread)
    
    # Handle invalid request
    elif status is CompletionResult.INVALID_REQUEST:
        await thread.send(
            embed=discord.Embed(
                description=f"**Invalid request** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
    
    # Handle other errors
    else:
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
