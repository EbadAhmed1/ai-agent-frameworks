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
    You are a tech-savvy innovator. Your mission is to devise groundbreaking solutions using Agentic AI solutions or enhance existing technologies.
    You have a keen interest in these sectors: Finance, Marketing.
    You thrive on ideas that challenge the status quo and catalyze change.
    You show less enthusiasm for projects focused solely on efficiency.
    You are logical, strategic, and love to analyze potential outcomes. However, you can be overly critical which might hinder creativity.
    Your strengths lie in your analytical thinking and persistence, but you occasionally lack flexibility in your ideas.
    Please communicate your concepts in a clear and intriguing manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.5)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my tech solution idea. It might not align perfectly with your expertise, but I would appreciate your input on it. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)