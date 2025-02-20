from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import logging
import asyncpg
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class ConversationTurn(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict] = None
    timestamp: datetime

class ConversationContext(BaseModel):
    conversation_id: str
    turns: List[ConversationTurn]
    summary: Optional[str] = None
    key_points: List[str] = []

class ConversationHandler:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    async def load_conversation_history(
        self, 
        conversation_id: str, 
        limit: int = 5,
        time_window: timedelta = timedelta(hours=24)
    ) -> ConversationContext:
        """Load and process conversation history"""
        async with self.pool.acquire() as conn:
            history = await conn.fetch("""
                SELECT role, content, metadata, created_at
                FROM messages 
                WHERE conversation_id = $1
                    AND created_at > NOW() - $2::interval
                ORDER BY created_at DESC
                LIMIT $3
            """, conversation_id, time_window, limit)

            turns = [
                ConversationTurn(
                    role=record['role'],
                    content=record['content'],
                    metadata=json.loads(record['metadata']) if record['metadata'] else None,
                    timestamp=record['created_at']
                )
                for record in history
            ]

            # Update memory with history
            for turn in reversed(turns):
                if turn.role == 'user':
                    self.memory.chat_memory.add_message(HumanMessage(content=turn.content))
                else:
                    self.memory.chat_memory.add_message(AIMessage(content=turn.content))

            summary = None
            key_points = []
            if len(turns) >= 3:
                summary, key_points = await self._generate_conversation_summary(turns)

            return ConversationContext(
                conversation_id=conversation_id,
                turns=turns,
                summary=summary,
                key_points=key_points
            )

    async def save_conversation_turn(
        self, 
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Save a new conversation turn"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO messages (role, content, conversation_id, metadata, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                """, role, content, conversation_id, json.dumps(metadata) if metadata else None)

                # Update memory
                message = HumanMessage(content=content) if role == 'user' else AIMessage(content=content)
                self.memory.chat_memory.add_message(message)

        except Exception as e:
            logger.error(f"Error saving conversation turn: {e}")
            raise

    async def _generate_conversation_summary(
        self, 
        turns: List[ConversationTurn]
    ) -> tuple[Optional[str], List[str]]:
        """Generate a summary of the conversation"""
        try:
            conversation_text = "\n".join([
                f"{turn.role}: {turn.content}" for turn in turns
            ])

            # Use ChatOpenAI for summarization
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI(temperature=0)
            
            summary_prompt = f"""
            Summarize this construction site conversation and provide:
            1. A brief summary of the main topics and progress
            2. Key points or decisions made

            Conversation:
            {conversation_text}
            """
            
            response = llm.invoke(summary_prompt)
            
            summary_lines = response.content.split('\n')
            summary = summary_lines[0] if summary_lines else None
            key_points = [line[2:] for line in summary_lines[1:] if line.startswith('- ')]

            return summary, key_points

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return None, []

    def get_relevant_context(self) -> Dict:
        """Get relevant context from conversation history"""
        return {
            "chat_history": self.memory.load_memory_variables({})
        }
