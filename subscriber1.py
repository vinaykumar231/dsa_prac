import redis

def main():
    r = redis.Redis(host="localhost", port=6379, db=0)
    pubsub = r.pubsub()
    pubsub.subscribe("stock_updates")

    print("📱 Customer App listening for stock updates...")
    for message in pubsub.listen():
        if message["type"] == "message":
            print(f"📱 Customer App received: {message['data'].decode()}")

if __name__ == "__main__":
    main()
