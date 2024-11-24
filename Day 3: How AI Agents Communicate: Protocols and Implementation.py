from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import uuid
import time
from queue import Queue

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"

@dataclass
class Message:
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Any
    conversation_id: str = None
    timestamp: float = None
    
    def __post_init__(self):
        if not self.conversation_id:
            self.conversation_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

class MessageBus:
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.subscribers: Dict[str, List[str]] = {}
    
    def register_agent(self, agent_id: str):
        """Register a new agent with the message bus"""
        if agent_id not in self.queues:
            self.queues[agent_id] = Queue()
            self.subscribers[agent_id] = []
    
    def send_message(self, message: Message):
        """Send a message to a specific agent"""
        if message.receiver_id in self.queues:
            self.queues[message.receiver_id].put(message)
    
    def broadcast(self, sender_id: str, content: Any):
        """Send a message to all subscribers"""
        message = Message(
            sender_id=sender_id,
            receiver_id="broadcast",
            message_type=MessageType.BROADCAST,
            content=content
        )
        for subscriber in self.subscribers.get(sender_id, []):
            self.queues[subscriber].put(message)
    
    def subscribe(self, subscriber_id: str, publisher_id: str):
        """Subscribe to broadcasts from a specific agent"""
        if publisher_id in self.subscribers:
            self.subscribers[publisher_id].append(subscriber_id)

class Agent:
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.message_bus.register_agent(self.agent_id)
        self.conversation_history: Dict[str, List[Message]] = {}
    
    def send_message(self, to_agent: str, content: Any, 
                    msg_type: MessageType = MessageType.REQUEST):
        """Send a message to another agent"""
        message = Message(
            sender_id=self.agent_id,
            receiver_id=to_agent,
            message_type=msg_type,
            content=content
        )
        self.message_bus.send_message(message)
        # Store in conversation history
        if message.conversation_id not in self.conversation_history:
            self.conversation_history[message.conversation_id] = []
        self.conversation_history[message.conversation_id].append(message)
    
    def broadcast(self, content: Any):
        """Broadcast a message to all subscribers"""
        self.message_bus.broadcast(self.agent_id, content)
    
    def receive_messages(self) -> List[Message]:
        """Receive all pending messages"""
        messages = []
        queue = self.message_bus.queues[self.agent_id]
        while not queue.empty():
            message = queue.get()
            messages.append(message)
            # Store in conversation history
            if message.conversation_id not in self.conversation_history:
                self.conversation_history[message.conversation_id] = []
            self.conversation_history[message.conversation_id].append(message)
        return messages

# Let's test it out!
def run_simple_conversation():
    # Create message bus
    bus = MessageBus()
    
    # Create agents
    alice = Agent("alice", bus)
    bob = Agent("bob", bus)
    
    # Subscribe to broadcasts
    bus.subscribe("bob", "alice")
    
    # Alice sends a message to Bob
    alice.send_message("bob", "Hey Bob, what's the weather like?")
    
    # Bob receives and responds
    messages = bob.receive_messages()
    for msg in messages:
        print(f"Bob received: {msg.content}")
        bob.send_message(
            "alice", 
            "It's sunny here!",
            MessageType.RESPONSE
        )
    
    # Alice receives Bob's response
    messages = alice.receive_messages()
    for msg in messages:
        print(f"Alice received: {msg.content}")
    
    # Alice broadcasts a message
    alice.broadcast("Going offline for a bit!")
    
    # Bob receives the broadcast
    messages = bob.receive_messages()
    for msg in messages:
        print(f"Bob received broadcast: {msg.content}")

if __name__ == "__main__":
    run_simple_conversation()
