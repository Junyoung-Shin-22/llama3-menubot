import discord
from llama3_chat import llama_single_inference, SYSTEM

from datetime import datetime, timezone, timedelta
from threading import Lock

with open('./token.txt') as f:
    TOKEN = f.read()
with open('./channels.txt') as f:
    CHANNELS = list(map(int, 
                        f.read().splitlines()))

_INTENT = discord.Intents.all()
BOT = discord.Client(intents=_INTENT)

@BOT.event
async def on_ready():
    print(f'{BOT.user} is ready')

 
class Conversation:
    instances = dict()

    def __init__(self, message):
        self.lock = Lock()

        self.init_message = message
        self.messages = [SYSTEM, ]

        Conversation.instances[message.id] = self

    def get_conversation_by_user(user):
        users_to_end = []
        conv_to_return = None

        for conv in Conversation.instances.values():
            # if conversation is too old, end it
            if datetime.now(timezone.utc) - conv.init_message.created_at > timedelta(minutes=3):
                users_to_end.append(conv.init_message.author)
                continue

            if conv.init_message.author == user:
                conv_to_return = conv
        
        for user in users_to_end:
            Conversation.end_conversation_with_user(user)
        return conv_to_return

    def end_conversation_with_user(user):
        keys_to_end = []

        for key, conv in Conversation.instances.items():
            if conv.init_message.author == user:
                keys_to_end.append(key)
        
        for key in keys_to_end:
            Conversation.instances.pop(key, None)

    
    def add_user_content(self, content):
        self.messages.append({'role': 'user', 'content': content})

    def add_bot_content(self):
        response = llama_single_inference(self.messages)
        self.messages.append({'role': 'assistant', 'content': response})

@BOT.event
async def on_message(message):
    return_cond = [
        message.author.id == BOT.user.id,
        message.channel.id not in CHANNELS,
    ]
    if any(return_cond):
        return

    conv = Conversation.get_conversation_by_user(message.author)

    if conv is None: # conversation not exists
        if message.content == 'ㅈㅁㅁ':
            conv = Conversation(message)
        else:
            return
    elif message.content == 'ㅇㅋ': # user wants to end conversation
        Conversation.end_conversation_with_user(message.author)
        message.channel.send('굳')
        return
    
    # do something with the conversation
    await do_conversation(conv, message)

        

async def do_conversation(conv, message):
    with conv.lock:
        conv.add_user_content(message.content)
        conv.add_bot_content()
        
        response = conv.messages[-1]['content']
    await message.channel.send(response)


if __name__ == '__main__':
    BOT.run(TOKEN)