import asyncio

class BiraMediator:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.handlers = {}
        self.name = "mediator"

    def register(self, name, handler):
        self.handlers[name] = handler

    # Broadcast
    async def send(self, sender, message):
        if isinstance(message, str):
            message = {message: None}
        await self.queue.put(("broadcast", sender.name, None, message))

    # One-to-one
    async def send_to(self, sender, target, message):
        await self.queue.put(("direct", sender, target, message))

    async def run(self):
        while True:
            msg_type, sender, target, message = await self.queue.get()
            print("MEDIATOR GOT:", msg_type, message)
            
            if msg_type == "broadcast":
                for name, handler in self.handlers.items():
                    if name != sender:
                        await handler.receive(message)

            elif msg_type == "direct":
                if target in self.handlers:
                    await self.handlers[target].receive(message)

