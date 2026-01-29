class BiraComponent:
    def __init__(self, name, mediator):
        self.name = name
        self.mediator = mediator
        mediator.register(name, self)

    async def send(self, message):
        await self.mediator.send(self.name, message)
        
    async def send_to(self, sender, target, message):
        await self.mediator.send_to(sender, target, message)

    async def receive(self, message):
        print(self.name)
        raise NotImplementedError(f"{self.name}: Subclasses must implement this method")