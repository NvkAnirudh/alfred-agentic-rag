import os
from dotenv import load_dotenv
load_dotenv()

import datasets
import asyncio
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        text="\n".join([
            f"Name: {guest_dataset['name'][i]}",
            f"Relation: {guest_dataset['relation'][i]}",
            f"Description: {guest_dataset['description'][i]}",
            f"Email: {guest_dataset['email'][i]}"
        ]),
        metadata={"name": guest_dataset['name'][i]}
    )
    for i in range(len(guest_dataset))
]

bm25_retirever = BM25Retriever.from_defaults(nodes=docs)

def get_guest_info_retirever(query: str):
    """
    Retrieves detailed information about gala guests based on their name or relation
    """
    results = bm25_retirever.retrieve(query)
    if results:
        return "\n\n".join([doc.text for doc in results])
    else:
        return "No relevant documents found"

# Initialize the tool
guest_info_tool = FunctionTool.from_defaults(get_guest_info_retirever)


# Initialize the OpenAI model
llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

alfred = AgentWorkflow.from_tools_or_functions([guest_info_tool], llm=llm)

async def main(): 
    response = await alfred.run("Who is Ada Lovelace?")
    print("Alfred's response:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
