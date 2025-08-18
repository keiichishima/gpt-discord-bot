# Base data structures for Discord bot
# Contains core data classes and constants used throughout the application

from dataclasses import dataclass
from typing import Optional, List

# Token used to separate different contexts or conversations
SEPARATOR_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class Message:
    """Immutable message data structure
    
    Represents a single message in a conversation with user attribution
    and optional text content. Used internally for conversation history.
    """
    user: str                           # Username who sent the message
    text: Optional[str] = None          # Message content (None for system messages)

    def render(self) -> str:
        """Render the message in a human-readable format
        
        Creates a standardized string representation of the message
        in the format "username: message text" for AI processing.
        
        Returns:
            Formatted string representation of the message
        """
        result = self.user + ":"
        if self.text is not None:
            result += " " + self.text
        return result


@dataclass(frozen=True)
class Config:
    """Configuration data structure for bot settings
    
    Stores bot configuration including name and system behavior.
    Immutable to prevent accidental modification during runtime.
    """
    name: str                          # Bot name identifier
    system_message: str                # System prompt for AI behavior
