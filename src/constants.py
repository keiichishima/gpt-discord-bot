# Configuration constants and settings for Discord bot
# Loads environment variables, configuration files, and defines system constants

import os
import dacite
import yaml
from typing import Dict, List
from dotenv import load_dotenv

from src.base import Config

# Load environment variables from .env file
load_dotenv()

# Load bot configuration from YAML file
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG: Config = dacite.from_dict(
    Config, yaml.safe_load(open(os.path.join(SCRIPT_DIR, "config.yaml"), "r"))
)

# Bot identity and behavior configuration
BOT_NAME = CONFIG.name
# AI agent system prompt with tool usage instructions
SYSTEM_MESSAGE = CONFIG.system_message + '''
You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOLS_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Actions:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Actions/Observation until you reach a conclusion)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Use tools if necessary.
Respond directly if appropriate.
Format is Action:```$JSON_BLOB```then Observation
'''

# Human message template for agent input formatting
HUMAN_MESSAGE = '''
{input}

{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)
'''

# API credentials and configuration from environment variables
DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]      # Discord bot authentication token
DISCORD_CLIENT_ID = os.environ["DISCORD_CLIENT_ID"]      # Discord application client ID
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]            # OpenAI API key for AI responses
OPENAI_MODEL = os.environ["OPENAI_MODEL"]                # OpenAI model to use (e.g., gpt-4)

# Server access control - parse comma-separated server IDs from environment
ALLOWED_SERVER_IDS: List[int] = []
server_ids = os.environ["ALLOWED_SERVER_IDS"].split(",")
for s in server_ids:
    ALLOWED_SERVER_IDS.append(int(s))

# Moderation channel mapping - maps server ID to moderation channel ID
SERVER_TO_MODERATION_CHANNEL: Dict[int, int] = {}
server_channels = os.environ.get("SERVER_TO_MODERATION_CHANNEL", "").split(",")
for s in server_channels:
    if ":" in s:  # Format: "server_id:channel_id"
        values = s.split(":")
        SERVER_TO_MODERATION_CHANNEL[int(values[0])] = int(values[1])

# Discord bot invite URL with required permissions
# Permissions: Send Messages, Create Public Threads, Send Messages in Threads, 
# Manage Messages, Manage Threads, Read Message History, Use Slash Commands
BOT_INVITE_URL = f"https://discord.com/api/oauth2/authorize?client_id={DISCORD_CLIENT_ID}&permissions=328565073920&scope=bot"

# Content moderation thresholds for blocking content (stricter thresholds)
# Content exceeding these scores will be completely blocked and deleted
MODERATION_VALUES_FOR_BLOCKED = {
    "hate": 0.5,                    # Hate speech threshold
    "hate/threatening": 0.1,        # Threatening hate speech (very low tolerance)
    "self-harm": 0.2,              # Self-harm content
    "sexual": 0.5,                 # Sexual content
    "sexual/minors": 0.2,          # Sexual content involving minors (strict)
    "violence": 0.7,               # Violence content
    "violence/graphic": 0.8,       # Graphic violence content
}

# Content moderation thresholds for flagging content (lenient thresholds)
# Content exceeding these scores will be flagged but not blocked
MODERATION_VALUES_FOR_FLAGGED = {
    "hate": 0.4,                   # Hate speech flagging threshold
    "hate/threatening": 0.05,      # Threatening hate speech flagging (very sensitive)
    "self-harm": 0.1,             # Self-harm content flagging
    "sexual": 0.3,                # Sexual content flagging
    "sexual/minors": 0.1,         # Sexual content involving minors flagging (very sensitive)
    "violence": 0.1,              # Violence content flagging
    "violence/graphic": 0.1,      # Graphic violence content flagging
}

# Bot behavior and Discord limitations configuration

# Message processing delay to handle rapid user messages
SECONDS_DELAY_RECEIVING_MSG = 2  # Wait 2 seconds to catch multiple consecutive messages

# Thread management settings
MAX_THREAD_MESSAGES = 20                    # Maximum messages before closing thread
ACTIVATE_THREAD_PREFX = "💬✅"              # Prefix for active conversation threads
INACTIVATE_THREAD_PREFIX = "💬❌"           # Prefix for closed/inactive threads

# Discord message size limits
MAX_CHARS_PER_REPLY_MSG = 1500             # Split messages at 1.5k chars (Discord limit is 2k)
