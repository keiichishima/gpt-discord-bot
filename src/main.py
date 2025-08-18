# Discord bot main module
# Handles Discord client initialization, commands, and message processing

import discord
from discord import Message as DiscordMessage
import logging
import asyncio

# Internal imports
from src.base import Message
from src.constants import (
    BOT_INVITE_URL,
    DISCORD_BOT_TOKEN,
    ACTIVATE_THREAD_PREFX,
    MAX_THREAD_MESSAGES,
    SECONDS_DELAY_RECEIVING_MSG,
)
from src.utils import (
    logger,
    should_block,
    close_thread,
    is_last_message_stale,
    discord_message_to_message,
)
from src import completion
from src.completion import generate_completion_response, process_response
from src.moderation import (
    moderate_message,
    send_moderation_blocked_message,
    send_moderation_flagged_message,
)

# Configure logging format and level
logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)

# Set up Discord client with required intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content

# Initialize Discord client and command tree
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)


@client.event
async def on_ready():
    """Event handler for when the bot successfully connects to Discord"""
    logger.info(f"We have logged in as {client.user}. Invite URL: {BOT_INVITE_URL}")
    # Set the bot name for completion module
    completion.MY_BOT_NAME = client.user.name
    # Sync command tree to make slash commands available
    await tree.sync()


# Slash command: /chat
@tree.command(name="chat", description="Create a new thread for conversation")
@discord.app_commands.checks.has_permissions(send_messages=True)
@discord.app_commands.checks.has_permissions(view_channel=True)
@discord.app_commands.checks.bot_has_permissions(send_messages=True)
@discord.app_commands.checks.bot_has_permissions(view_channel=True)
@discord.app_commands.checks.bot_has_permissions(manage_threads=True)
async def chat_command(int: discord.Interaction, message: str):
    """Slash command handler for creating new chat threads
    
    Args:
        int: Discord interaction object
        message: User's initial message to start the conversation
    """
    try:
        # Only support creating threads in text channels
        if not isinstance(int.channel, discord.TextChannel):
            return

        # Block servers not in allow list
        if should_block(guild=int.guild):
            return

        user = int.user
        logger.info(f"Chat command by {user} {message[:20]}")
        
        try:
            # Moderate the initial message
            flagged_str, blocked_str = moderate_message(message=message, user=user)
            await send_moderation_blocked_message(
                guild=int.guild,
                user=user,
                blocked_str=blocked_str,
                message=message,
            )
            
            # Handle blocked messages
            if len(blocked_str) > 0:
                await int.response.send_message(
                    f"Your prompt has been blocked by moderation.\n{message}",
                    ephemeral=True,
                )
                return

            # Create embed for the chat initiation
            embed = discord.Embed(
                description=f"<@{user.id}> wants to chat! 🤖💬",
                color=discord.Color.green(),
            )
            embed.add_field(name=user.name, value=message)

            # Handle flagged messages
            if len(flagged_str) > 0:
                embed.color = discord.Color.yellow()
                embed.title = "⚠️ This prompt was flagged by moderation."

            # Send initial response
            await int.response.send_message(embed=embed)
            response = await int.original_response()

            # Send moderation notification if flagged
            await send_moderation_flagged_message(
                guild=int.guild,
                user=user,
                flagged_str=flagged_str,
                message=message,
                url=response.jump_url,
            )
        except Exception as e:
            logger.exception(e)
            await int.response.send_message(
                f"Failed to start chat {str(e)}", ephemeral=True
            )
            return

        # Create the conversation thread
        thread = await response.create_thread(
            name=f"{ACTIVATE_THREAD_PREFX} {user.name[:20]} - {message[:30]}",
            slowmode_delay=1,  # Prevent spam
            reason="gpt-bot",
            auto_archive_duration=60,  # Archive after 1 hour of inactivity
        )
        
        # Generate and send AI response
        async with thread.typing():
            # Create message object for completion
            messages = [Message(user=user.name, text=message)]
            response_data = await generate_completion_response(
                messages=messages, user=user
            )
            # Process and send the response
            await process_response(
                user=user, thread=thread, response_data=response_data
            )
    except Exception as e:
        logger.exception(e)
        await int.response.send_message(
            f"Failed to start chat {str(e)}", ephemeral=True
        )


# Message event handler - processes all incoming messages
@client.event
async def on_message(message: DiscordMessage):
    """Event handler for all Discord messages
    
    Processes messages in bot-created threads and mentions in regular channels.
    Handles moderation, conversation history, and AI response generation.
    
    Args:
        message: The Discord message object
    """
    try:
        # Block servers not in allow list
        if should_block(guild=message.guild):
            return

        # Ignore messages from the bot itself
        if message.author == client.user:
            return

        # Determine message context
        channel = message.channel
        is_thread = isinstance(channel, discord.Thread)
        is_mentioned = client.user.mentioned_in(message)

        # Handle thread messages
        if is_thread:
            # Only process threads created by this bot
            if channel.owner_id != client.user.id:
                return

            # Skip archived, locked, or incorrectly named threads
            if (
                channel.archived
                or channel.locked
                or not channel.name.startswith(ACTIVATE_THREAD_PREFX)
            ):
                return

            # Close threads that exceed message limit
            if channel.message_count > MAX_THREAD_MESSAGES:
                await close_thread(thread=channel)
                return
        elif not is_mentioned:
            # Ignore non-thread messages where bot isn't mentioned
            return

        # Moderate incoming message
        flagged_str, blocked_str = moderate_message(
            message=message.content, user=message.author
        )
        await send_moderation_blocked_message(
            guild=message.guild,
            user=message.author,
            blocked_str=blocked_str,
            message=message.content,
        )
        
        # Handle blocked messages
        if len(blocked_str) > 0:
            try:
                # Try to delete the blocked message
                await message.delete()
                await channel.send(
                    embed=discord.Embed(
                        description=f"❌ **{message.author}'s message has been deleted by moderation.**",
                        color=discord.Color.red(),
                    )
                )
                return
            except Exception as e:
                # Handle case where bot lacks delete permissions
                await channel.send(
                    embed=discord.Embed(
                        description=f"❌ **{message.author}'s message has been blocked by moderation but could not be deleted. Missing Manage Messages permission in this Channel.**",
                        color=discord.Color.red(),
                    )
                )
                return
        
        # Handle flagged messages
        await send_moderation_flagged_message(
            guild=message.guild,
            user=message.author,
            flagged_str=flagged_str,
            message=message.content,
            url=message.jump_url,
        )
        if len(flagged_str) > 0:
            await channel.send(
                embed=discord.Embed(
                    description=f"⚠️ **{message.author}'s message has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )

        # Wait for potential follow-up messages to avoid processing duplicates
        if SECONDS_DELAY_RECEIVING_MSG > 0:
            await asyncio.sleep(SECONDS_DELAY_RECEIVING_MSG)
            if is_last_message_stale(
                interaction_message=message,
                last_message=channel.last_message,
                bot_id=client.user.id,
            ):
                # Skip if there's a newer message
                return

        logger.info(
            f"Thread message to process - {message.author}: {message.content[:50]} - {channel.name} {channel.jump_url}"
        )

        # Retrieve conversation history
        channel_messages = [
            discord_message_to_message(message)
            async for message in channel.history(limit=MAX_THREAD_MESSAGES)
        ]
        # Filter out None values and reverse to chronological order
        channel_messages = [x for x in channel_messages if x is not None]
        channel_messages.reverse()

        # Generate AI response
        async with channel.typing():
            response_data = await generate_completion_response(
                messages=channel_messages, user=message.author
            )

        # Check if message is still the latest before responding
        if is_last_message_stale(
            interaction_message=message,
            last_message=channel.last_message,
            bot_id=client.user.id,
        ):
            # Skip if there's a newer message from a user
            return

        # Process and send the AI response
        await process_response(
            user=message.author, thread=channel, response_data=response_data
        )
    except Exception as e:
        logger.exception(e)


# Start the Discord bot
client.run(DISCORD_BOT_TOKEN)
