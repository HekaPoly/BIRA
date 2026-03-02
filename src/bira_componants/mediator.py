import asyncio
from queue import Queue
from time import sleep

class BiraMediator:
    def __init__(self):
        self.queue = Queue(maxsize=0)
        self.handlers = {}
        self.name = "mediator"

    def register(self, name, handler):
        self.handlers[name] = handler

    # Broadcast
    def send(self, sender, message):
        if isinstance(message, str):
            message = {message: None}
        self.queue.put(("broadcast", sender.name, None, message))

    # One-to-one
    def send_to(self, sender, target, message):
        self.queue.put(("direct", sender, target, message))
    def clear (self): 
        with self.queue.mutex: 
            self.queue.queue.clear()
            
    def run(self):
        while True:
            msg_type, sender, target, message = self.queue.get()
            
            if msg_type == "broadcast":
                print("MEDIATOR GOT:", msg_type, message)

                for name, handler in self.handlers.items():
                    print(name, " has ", message)
                    handler.receive(message)

            elif msg_type == "direct":
                if target in self.handlers:
                    print("SENDING DIRECT MESSAGE :", target)
                    self.handlers[target].receive(message)
            
            sleep(0.5)

