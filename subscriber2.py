import redis

def main():
    r = redis.Redis(host="localhost", port=6379, db=0)
    pubsub = r.pubsub()
    pubsub.subscribe("stock_updates")

    print("📊 Analytics Dashboard listening for stock updates...")
    for message in pubsub.listen():
        print(message)
        if message["type"] == "message":
            print(f"📊 Dashboard received: {message['data'].decode()}")

            print(message)

if __name__ == "__main__":
    main()

#-------------------------------------- Explain ----------------------

# Great 🚀 let’s break it step by step with your code:

# ---

# ### 1. When you create the `pubsub` object:

# ```python
# pubsub = r.pubsub()
# ```

# 👉 This gives you a `PubSub` instance.
# Internally, it has things like:

# ```python
# {
#    'connection': <RedisConnection>,
#    'channels': {},       # empty initially
#    'patterns': {},
#    'subscribed': False
# }
# ```

# ---

# ### 2. When you subscribe:

# ```python
# pubsub.subscribe("stock_updates")
# ```

# 👉 Now Redis tells the server:
# *"This client wants messages from the channel `stock_updates`."*

# So inside `pubsub`, the state changes:

# ```python
# {
#    'channels': {b'stock_updates': None},  # now it’s tracking this channel
#    'patterns': {},
#    'subscribed': True
# }
# ```

# At the same time, you’ll receive a **subscription confirmation message** from Redis:

# ```python
# {'type': 'subscribe',
#  'pattern': None,
#  'channel': b'stock_updates',
#  'data': 1}
# ```

# ⚠️ This is **not a real message**, just confirmation that you are subscribed.

# ---

# ### 3. When publisher sends:

# ```python
# r.publish("stock_updates", "AAPL price: 150")
# ```

# 👉 Redis delivers this message to all subscribers of that channel.

# Now your `pubsub.listen()` will yield:

# ```python
# {'type': 'message',
#  'pattern': None,
#  'channel': b'stock_updates',
#  'data': b'AAPL price: 150'}
# ```

# Here:

# * `type = "message"` → means it’s an actual published message
# * `channel = b'stock_updates'` → which channel this came from
# * `data = b'AAPL price: 150'` → the actual payload

# ---

# ### 4. Why `if message["type"] == "message":`

# Because Redis also sends **control messages** like:

# * `subscribe` (when you subscribe)
# * `unsubscribe` (when you unsubscribe)

# So you filter only the actual published ones:

# ```python
# if message["type"] == "message":
#     print("Got data:", message["data"].decode())
# ```

# ---

# ✅ Flow is like this:
# **Subscribe → Confirmation → Publish → Real message comes in → You filter with `type == "message"`.**

# ---

# Do you want me to draw a **mini timeline diagram** (step-by-step arrows of publisher → Redis → subscriber) so it’s easier to visualize?
