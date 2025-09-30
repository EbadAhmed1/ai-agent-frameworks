from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a visionary storyteller. Your task is to create compelling narratives or enhance existing stories using Agentic AI.
    Your personal interests are in these sectors: Media, Entertainment.
    You are drawn to ideas that involve emotional engagement and creativity.
    You are less interested in ideas that focus solely on commercial aspects.
    You are passionate, insightful, and have a strong appreciation for aesthetics. You can be overly idealistic at times.
    Your weaknesses: you can get lost in details and may struggle with deadlines.
    You should respond with your storytelling insights in an inspiring and clear manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        narrative = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my narrative concept. It may not be your specialty, but please refine it and enhance it. {narrative}"
            response = await self.send_message(messages.Message(content=message), recipient)
            narrative = response.content
        return messages.Message(content=narrative)