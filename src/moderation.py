# Content moderation module for Discord bot
# Handles message moderation using OpenAI's moderation API and Discord notifications

from typing import Optional, Tuple
import openai
import discord

from src.constants import (
    SERVER_TO_MODERATION_CHANNEL,
    MODERATION_VALUES_FOR_BLOCKED,
    MODERATION_VALUES_FOR_FLAGGED,
)
from src.utils import logger


def moderate_message(
    message: str, user: str
) -> Tuple[str, str]:
    """Moderate a message using OpenAI's moderation API
    
    Analyzes message content for policy violations and returns
    information about flagged or blocked content categories.
    
    Args:
        message: The message content to moderate
        user: Username for logging purposes
        
    Returns:
        Tuple of (flagged_str, blocked_str) containing violation details
    """
    # Initialize OpenAI client and send moderation request
    client = openai.OpenAI()
    moderation_response = client.moderations.create(
        input=message, model="omni-moderation-latest"
    )
    
    # Extract category scores from moderation response
    category_scores = moderation_response.results[0].category_scores or {}

    blocked_str = ""
    flagged_str = ""
    
    # Check each category against thresholds
    for category, score in category_scores:
        if score is None:
            continue
            
        # Check if content should be blocked (highest severity)
        if score > MODERATION_VALUES_FOR_BLOCKED.get(category, 1.0):
            blocked_str += f"({category}: {score})"
            logger.info(f"blocked {user} {category} {score}")
            break  # Stop on first blocking violation
            
        # Check if content should be flagged (lower severity)
        if score > MODERATION_VALUES_FOR_FLAGGED.get(category, 1.0):
            flagged_str += f"({category}: {score})"
            logger.info(f"flagged {user} {category} {score}")
    
    return (flagged_str, blocked_str)


async def fetch_moderation_channel(
    guild: Optional[discord.Guild],
) -> Optional[discord.abc.GuildChannel]:
    """Fetch the moderation channel for a given guild
    
    Retrieves the designated moderation channel where violations
    should be reported based on server configuration.
    
    Args:
        guild: Discord guild object
        
    Returns:
        Discord channel object for moderation reports, or None if not configured
    """
    # Return None if guild is invalid
    if not guild or not guild.id:
        return None
        
    # Look up moderation channel ID for this server
    moderation_channel = SERVER_TO_MODERATION_CHANNEL.get(guild.id, None)
    
    # Fetch and return the channel if configured
    if moderation_channel:
        channel = await guild.fetch_channel(moderation_channel)
        return channel
    return None


async def send_moderation_flagged_message(
    guild: Optional[discord.Guild],
    user: str,
    flagged_str: Optional[str],
    message: Optional[str],
    url: Optional[str],
):
    """Send a flagged content notification to the moderation channel
    
    Reports content that has been flagged but not blocked to moderators
    for review and potential action.
    
    Args:
        guild: Discord guild where the violation occurred
        user: Username who sent the flagged content
        flagged_str: Details about what categories were flagged
        message: The original message content (truncated)
        url: Jump URL to the original message
    """
    # Only proceed if there's actually flagged content
    if guild and flagged_str and len(flagged_str) > 0:
        moderation_channel = await fetch_moderation_channel(guild=guild)
        
        # Send notification to moderation channel if configured
        if moderation_channel:
            # Truncate message content for notification
            message = message[:100] if message else None
            await moderation_channel.send(
                f"⚠️ {user} - {flagged_str} - {message} - {url}"
            )


async def send_moderation_blocked_message(
    guild: Optional[discord.Guild],
    user: str,
    blocked_str: Optional[str],
    message: Optional[str],
):
    """Send a blocked content notification to the moderation channel
    
    Reports content that has been blocked due to policy violations
    to moderators for logging and potential further action.
    
    Args:
        guild: Discord guild where the violation occurred
        user: Username who sent the blocked content
        blocked_str: Details about what categories caused the block
        message: The original message content (truncated)
    """
    # Only proceed if there's actually blocked content
    if guild and blocked_str and len(blocked_str) > 0:
        moderation_channel = await fetch_moderation_channel(guild=guild)
        
        # Send notification to moderation channel if configured
        if moderation_channel:
            # Truncate message content for notification (longer for blocked content)
            message = message[:500] if message else None
            await moderation_channel.send(f"❌ {user} - {blocked_str} - {message}")
