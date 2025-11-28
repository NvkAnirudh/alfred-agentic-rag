import os
from dotenv import load_dotenv
load_dotenv()

import datasets
from smolagents import Tool, CodeAgent, InferenceClientModel, ToolCallingAgent, OpenAIServerModel
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content = "\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}"
            f"Email: {guest['email']}"
        ]),
        metadata = {"name": guest['name']}
    )
    for guest in guest_dataset
]

class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves information about a guest by name or relation"
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about"
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No relevant documents found"

# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)

# Intialize the Hugging face model
# model = InferenceClientModel()

# Faster alternatives
# model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct-Turbo")

model = OpenAIServerModel(model_id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Create Alfred with the guest info tool
# alfred = CodeAgent(tools=[guest_info_tool], model=model)

# Create Alfred with the guest info tool
alfred = ToolCallingAgent(tools=[guest_info_tool], model=model, instructions="""You are Alfred, a helpful assistant at a gala event.
When asked about guests, use the guest_info_retriever tool and then provide 
a natural, conversational response based on the information retrieved.
Do not just dump the raw data - synthesize it into a friendly answer.""", stream_outputs=True)

# Example query Alfred might receive during the gala
response = alfred.run("Who is Ada Lovelace?")
print("Alfred's response:")
for chunk in response:
    print(chunk, end="", flush=True)
