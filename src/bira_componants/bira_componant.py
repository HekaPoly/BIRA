class BiraComponent:
    def __init__(self, name, mediator):
        self.name = name
        self.mediator = mediator
        mediator.register(name, self)

    def send(self, message):
        self.mediator.send(self.name, message)
        
    def send_to(self, sender, target, message):
        self.mediator.send_to(sender, target, message)

    def receive(self, message):
        print(self.name)
        raise NotImplementedError(f"{self.name}: Subclasses must implement this method")