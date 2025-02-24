from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple
from langchain.schema import BaseMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings

class WeightedConversationMemory(ConversationBufferMemory):
    def __init__(
        self,
        decay_rate: float = 0.1,  # How quickly older messages lose importance
        max_token_limit: int = 2000,  # Limit total tokens in memory
        time_weight_factor: float = 0.5,  # How much to consider time decay
        relevance_weight_factor: float = 0.5,  # How much to consider semantic relevance
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.decay_rate = decay_rate
        self.max_token_limit = max_token_limit
        self.time_weight_factor = time_weight_factor
        self.relevance_weight_factor = relevance_weight_factor
        self.embedding_model = OpenAIEmbeddings()

    async def get_weighted_history(self, current_query: str) -> List[BaseMessage]:
        if not self.chat_memory.messages:
            return []

        weighted_messages = []
        current_time = datetime.now()
        query_embedding = await self.embedding_model.aembed_query(current_query)

        for msg in self.chat_memory.messages:
            # Calculate time decay
            message_time = msg.metadata.get('timestamp', current_time)
            hours_old = (current_time - message_time).total_seconds() / 3600
            time_weight = np.exp(-self.decay_rate * hours_old)

            # Calculate semantic relevance
            message_embedding = await self.embedding_model.aembed_query(msg.content)
            relevance_score = np.dot(query_embedding, message_embedding)

            # Combine weights
            final_weight = (
                self.time_weight_factor * time_weight +
                self.relevance_weight_factor * relevance_score
            )

            weighted_messages.append((msg, final_weight))

        # Sort by weight and apply token limit
        weighted_messages.sort(key=lambda x: x[1], reverse=True)
        filtered_messages = self._apply_token_limit(weighted_messages)

        return [msg for msg, _ in filtered_messages]

    def _apply_token_limit(self, weighted_messages: List[Tuple[BaseMessage, float]]) -> List[Tuple[BaseMessage, float]]:
        total_tokens = 0
        filtered_messages = []

        for msg, weight in weighted_messages:
            # Rough token estimation
            estimated_tokens = len(msg.content.split()) * 1.3
            if total_tokens + estimated_tokens <= self.max_token_limit:
                filtered_messages.append((msg, weight))
                total_tokens += estimated_tokens
            else:
                break

        return filtered_messages
