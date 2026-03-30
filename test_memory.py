import logging
logging.basicConfig(level=logging.INFO)

from memory_manager import MemoryManager

def run_test():
    mem = MemoryManager()
    
    # Wait a sec for background init
    import time
    time.sleep(2)
    
    print("\n--- TEST: RETRIEVAL TRIGGER ---")
    query = "Who was the guy that helped me set up the Orion architecture?"
    print(f"Query: {query}")
    ctx = mem.get_context_for_prompt(query)
    print("Gathered Context:")
    print(ctx)

    print("\n--- TEST: WRITE BACK ---")
    user_msg = "I was super stressed about my math exam today, but Alex helped me study at the library. We used the Pomodoro technique, which I really loved."
    assistant_msg = "I'm glad Alex could help! Pomodoro is great for math."
    print("Simulating conversation...")
    mem._background_store(user_msg, assistant_msg)
    
    time.sleep(3)
    print("Done testing.")

if __name__ == "__main__":
    run_test()
