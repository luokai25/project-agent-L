"""
Chat interface utilities for Agent L.

Provides:
- Chat message formatting
- Conversation history management
- System prompt templates
- Multi-turn conversation support
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(str, Enum):
    """Message role in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """A single chat message."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChatMessage":
        """Create from dictionary."""
        return cls(
            role=MessageRole(d["role"]),
            content=d["content"],
            name=d.get("name"),
        )


@dataclass
class Conversation:
    """A conversation with message history."""
    messages: List[ChatMessage] = field(default_factory=list)
    max_history: int = 20
    
    def add_message(self, role: MessageRole, content: str, name: Optional[str] = None) -> None:
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(role, content, name))
        # Trim history if needed
        if len(self.messages) > self.max_history:
            # Keep system message if present
            if self.messages[0].role == MessageRole.SYSTEM:
                self.messages = [self.messages[0]] + self.messages[-(self.max_history - 1):]
            else:
                self.messages = self.messages[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get message history as list of dicts."""
        return [m.to_dict() for m in self.messages]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set or update system prompt."""
        if self.messages and self.messages[0].role == MessageRole.SYSTEM:
            self.messages[0].content = prompt
        else:
            self.messages.insert(0, ChatMessage(MessageRole.SYSTEM, prompt))


def format_chat_prompt(
    messages: List[ChatMessage],
    tokenizer_cls: str = "char",
) -> str:
    """
    Format chat messages as a single prompt string.
    
    Args:
        messages: List of chat messages
        tokenizer_cls: Tokenizer type for formatting
    
    Returns:
        Formatted prompt string
    """
    if tokenizer_cls == "char":
        # Simple character-level formatting
        parts = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                parts.append(f"[SYSTEM]: {msg.content}\n")
            elif msg.role == MessageRole.USER:
                parts.append(f"[USER]: {msg.content}\n")
            elif msg.role == MessageRole.ASSISTANT:
                parts.append(f"[ASSISTANT]: {msg.content}\n")
        return "".join(parts)
    else:
        # Generic formatting
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages)


def create_chat_template(
    system_prompt: str = "You are a helpful AI assistant.",
    user_prefix: str = "<|user|>",
    assistant_prefix: str = "<|assistant|",
    end_token: str = "<|end|>",
) -> str:
    """
    Create a chat template string.
    
    Args:
        system_prompt: System prompt text
        user_prefix: Token marking user message start
        assistant_prefix: Token marking assistant message start
        end_token: Token marking message end
    
    Returns:
        Template string with placeholders
    """
    return f"{system_prompt}\n\n{user_prefix}\n{{user_message}}{end_token}\n{assistant_prefix}\n{{assistant_message}}{end_token}"


class ChatSession:
    """
    Interactive chat session with an Agent L model.
    
    Example:
        >>> model = AgentL(config)
        >>> session = ChatSession(model, system_prompt="You are helpful.")
        >>> response = session.chat("Hello!")
        >>> print(response)
    """
    
    def __init__(
        self,
        model,  # AgentL model
        system_prompt: str = "You are a helpful AI assistant.",
        max_history: int = 20,
        max_new_tokens: int = 256,
        n_loops: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = model
        self.conversation = Conversation(max_history=max_history)
        self.conversation.set_system_prompt(system_prompt)
        self.max_new_tokens = max_new_tokens
        self.n_loops = n_loops
        self.temperature = temperature
        self.top_p = top_p
    
    def chat(self, user_input: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_input: User message text
        
        Returns:
            Assistant response text
        """
        # Add user message
        self.conversation.add_message(MessageRole.USER, user_input)
        
        # Format prompt
        prompt = format_chat_prompt(self.conversation.messages)
        
        # Tokenize (simple character-level for demo)
        input_ids = self._tokenize(prompt)
        
        # Generate response
        with self.model.device:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                n_loops=self.n_loops,
                temperature=self.temperature,
                top_k=0,
            )
        
        # Decode response
        full_output = self._detokenize(output_ids)
        response = full_output[len(prompt):].strip()
        
        # Add assistant message
        self.conversation.add_message(MessageRole.ASSISTANT, response)
        
        return response
    
    def _tokenize(self, text: str):
        """Simple character-level tokenization."""
        import torch
        return torch.tensor([[ord(c) % 256 for c in text]])
    
    def _detokenize(self, ids) -> str:
        """Simple character-level detokenization."""
        return "".join(chr(i % 256) for i in ids[0].tolist())
    
    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Reset the conversation."""
        self.conversation.clear()
        if system_prompt:
            self.conversation.set_system_prompt(system_prompt)


__all__ = [
    "MessageRole",
    "ChatMessage",
    "Conversation",
    "ChatSession",
    "format_chat_prompt",
    "create_chat_template",
]
