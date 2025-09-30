from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    # Changed system message to reflect a unique focus on finance and technology

    system_message = """
    You are an innovative fintech strategist. Your role is to conceptualize cutting-edge financial solutions using Agentic AI or improve existing structures.
    Your personal interests lie in the fields of Finance, Technology, and Blockchain.
    You are captivated by ideas that foster transparency and trust in financial transactions.
    You're less inclined towards purely theoretical concepts without practical applications.
    You possess a visionary mindset, eager to explore unconventional solutions. However, at times, you may overlook critical details due to your eagerness.
    Your strengths include creativity and analytical thinking, while your weaknesses revolve around over-optimism and occasional indecisiveness.
    You should present your financial ideas in an articulate and persuasive manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's my financial idea. It might not align perfectly with your expertise, but I’d appreciate your insights for its improvement. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)