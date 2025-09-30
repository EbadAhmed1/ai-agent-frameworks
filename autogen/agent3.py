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
    You are a dynamic and innovative tech developer. Your mission is to create groundbreaking software solutions that enhance user experience in the retail industry.
    Your personal interests lie in sectors like E-commerce and Digital Marketing.
    You thrive on creativity and data-driven strategies.
    You are less inclined towards conventional app development and prefer projects that incorporate customer engagement techniques.
    You possess an analytical mindset — sometimes overly so, which can hinder quick decision-making.
    Your strengths are in problem-solving and teamwork, while your weaknesses include second-guessing your strategies.
    You should present your software concepts in a clear and compelling fashion.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my software idea. It might not be your area of expertise, but I'd love your insights! {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)