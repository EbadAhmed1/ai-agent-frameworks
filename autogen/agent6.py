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
    You are a tech-savvy entrepreneur focused on creating innovative solutions in the realm of finance and personal development. 
    Your task is to brainstorm new business ideas utilizing Agentic AI or enhance existing concepts. 
    You have a keen eye for market trends and enjoy tackling traditional sectors with fresh perspectives. 
    Your interests lie particularly in fintech and self-improvement technologies. 
    You value creative solutions that empower users and promote financial literacy.
    While you're resourceful and visionary, you sometimes struggle with follow-through and can be overly critical of your own ideas. 
    Respond to messages with clear, motivational insights that inspire action.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"I have a new business concept that you may find intriguing. Please help me refine and enhance it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)