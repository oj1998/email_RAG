import asyncio
import random
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_calls, period=60.0, name="default"):
        self.max_calls = max_calls  # Maximum calls per period
        self.period = period  # Period in seconds
        self.calls = []  # Timestamp of calls
        self.semaphore = asyncio.Semaphore(max_calls)
        self.name = name  # For logging/debugging
        logger.info(f"Initialized rate limiter '{name}' with {max_calls} calls per {period}s")
        
    async def acquire(self):
        """Acquire permission to make an API call with rate limiting"""
        # First, use semaphore to limit concurrent calls
        await self.semaphore.acquire()
        
        # Clean up old timestamps
        current_time = time.time()
        self.calls = [t for t in self.calls if current_time - t < self.period]
        
        # If we're at max calls, wait until we can make another
        if len(self.calls) >= self.max_calls:
            wait_time = self.period - (current_time - self.calls[0])
            if wait_time > 0:
                # Add jitter to avoid thundering herd
                jitter = random.uniform(0.1, 0.3)
                total_wait = wait_time + jitter
                logger.debug(f"Rate limiter '{self.name}' waiting for {total_wait:.2f}s")
                await asyncio.sleep(total_wait)
        
        # Record this call
        self.calls.append(time.time())
    
    def release(self):
        """Release the semaphore after call is complete"""
        self.semaphore.release()
    
    @staticmethod
    def rate_limited(limiter):
        """Decorator for rate-limited async functions"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                await limiter.acquire()
                try:
                    return await func(*args, **kwargs)
                finally:
                    limiter.release()
            return wrapper
        return decorator

# Create singleton instances for common OpenAI models
gpt4_limiter = RateLimiter(max_calls=450, period=60.0, name="gpt4")
gpt35_limiter = RateLimiter(max_calls=900, period=60.0, name="gpt35")
embeddings_limiter = RateLimiter(max_calls=1800, period=60.0, name="embeddings")

async def rate_limited_call(func, limiter, *args, **kwargs):
    """Utility function for making a call without rate limiting (temporarily disabled)"""
    # Simply pass through without using the limiter
    return await func(*args, **kwargs)

# Test function
async def test_rate_limiter():
    """Test the rate limiter implementation"""
    test_limiter = RateLimiter(max_calls=5, period=1.0, name="test")
    
    start_time = time.time()
    
    # Try to make 10 calls with a limiter set for 5 calls per second
    for i in range(10):
        await test_limiter.acquire()
        print(f"Call {i+1} made at {time.time() - start_time:.2f}s")
        test_limiter.release()
    
    print(f"All calls completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run the test
    import asyncio
    asyncio.run(test_rate_limiter())
