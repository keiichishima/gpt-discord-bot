# Utility functions for Discord bot operations
# Contains helper functions for message processing, thread management, and server access control

import logging
from typing import Optional, List
import discord
from discord import Message as DiscordMessage

from src.base import Message
from src.constants import (
    ALLOWED_SERVER_IDS,
    MAX_CHARS_PER_REPLY_MSG,
    INACTIVATE_THREAD_PREFIX,
)

# Initialize logger for this module
logger = logging.getLogger(__name__)


def discord_message_to_message(message: DiscordMessage) -> Optional[Message]:
    """Convert a Discord message to internal Message format
    
    Handles both thread starter messages (from embeds) and regular messages.
    Thread starter messages contain the original user input from slash commands.
    
    Args:
        message: Discord message object to convert
        
    Returns:
        Message object with user and text, or None if conversion fails
    """
    # Handle thread starter messages (from /chat command)
    if (
        message.type == discord.MessageType.thread_starter_message
        and message.reference.cached_message
        and len(message.reference.cached_message.embeds) > 0
        and len(message.reference.cached_message.embeds[0].fields) > 0
    ):
        # Extract user and message from embed field
        field = message.reference.cached_message.embeds[0].fields[0]
        if field.value:
            return Message(user=field.name, text=field.value)
    else:
        # Handle regular messages
        if message.content:
            return Message(user=message.author.name, text=message.content)
    
    return None


def split_into_shorter_messages(message: str) -> List[str]:
    """Split a long message into smaller chunks for Discord
    
    Discord has character limits for messages, so long AI responses
    need to be split into multiple messages to avoid errors.
    
    Args:
        message: The long message to split
        
    Returns:
        List of message chunks within Discord's character limits
    """
    return [
        message[i : i + MAX_CHARS_PER_REPLY_MSG]
        for i in range(0, len(message), MAX_CHARS_PER_REPLY_MSG)
    ]


def is_last_message_stale(
    interaction_message: DiscordMessage, last_message: DiscordMessage, bot_id: str
) -> bool:
    """Check if the interaction message is no longer the latest user message
    
    Prevents the bot from responding to old messages when users send
    multiple messages quickly. Only responds to the most recent user message.
    
    Args:
        interaction_message: The message that triggered the bot response
        last_message: The current last message in the channel
        bot_id: The bot's user ID to ignore bot messages
        
    Returns:
        True if the interaction message is stale (shouldn't respond)
    """
    return (
        last_message
        and last_message.id != interaction_message.id  # Different message
        and last_message.author
        and last_message.author.id != bot_id  # Not from the bot
    )


async def close_thread(thread: discord.Thread):
    """Close a Discord thread when it reaches limits or needs to be ended
    
    Updates the thread name, sends a closure notification, and archives
    the thread to prevent further messages.
    
    Args:
        thread: The Discord thread to close
    """
    # Update thread name to indicate it's inactive
    await thread.edit(name=INACTIVATE_THREAD_PREFIX)
    
    # Send closure notification to users
    await thread.send(
        embed=discord.Embed(
            description="**Thread closed** - Context limit reached, closing...",
            color=discord.Color.blue(),
        )
    )
    
    # Archive and lock the thread to prevent further activity
    await thread.edit(archived=True, locked=True)


def should_block(guild: Optional[discord.Guild]) -> bool:
    """Determine if the bot should block requests from a guild
    
    Implements server access control by checking against an allowlist
    of permitted servers. Also blocks direct messages.
    
    Args:
        guild: Discord guild object, None for DMs
        
    Returns:
        True if the request should be blocked
    """
    # Block direct messages (DMs not supported)
    if guild is None:
        logger.info(f"DM not supported")
        return True

    # Block servers not in the allowlist
    if guild.id and guild.id not in ALLOWED_SERVER_IDS:
        logger.info(f"Guild {guild} not allowed")
        return True
    
    return False
