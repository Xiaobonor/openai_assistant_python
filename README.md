# 🚀 OpenAI Assistant

### 🌐 Description

! This is an asynchronous Python package !  
OpenAI Assistant is a Python package designed to help developers seamlessly integrate OpenAI assistant capabilities into their applications.

## 📦 Installation

You can install the OpenAI Assistant package using pip:
    
```sh
pip install openai-assistant
```

## 🛠 Usage

You need to initialize this package with your OpenAI client.  
Example for Azure OpenAI:

```python
import asyncio
from openai_assistant import init, OpenAIAssistant
from openai import AsyncAzureOpenAI

# Initialize the OpenAI client with Azure endpoint and API key
openai = AsyncAzureOpenAI(
    azure_endpoint="https://openai.azure.com/",
    api_key="your-api-key",
    api_version="2024-05-01-preview"
)

# Initialize OpenAI assistants
assistants = init(openai)

# Define your agent or assistant class
class TestAgent(OpenAIAssistant):
    # Input your assistant ID; thread ID and callback are optional
    # If thread ID is not provided, a new thread will be generated automatically
    def __init__(self, callback=None):
        assistant_id = "your-assistant-id"
        super().__init__(assistant_id, None, callback)

    # Use this method to create an instance of your agent
    @classmethod
    async def create(cls, callback):
        self = cls(callback)
        await self.initialize_thread_id()
        return self

    # Define a method for submitting requests to the assistant
    async def submit_request(self, user_input: str):
        return await self.send_request(user_input)

# Create an agent instance and submit a request
agent = asyncio.run(TestAgent.create())
response = asyncio.run(agent.submit_request("Hello, how are you?"))
```

Want to enable your assistant to call custom functions? Register your function map:

```python
from openai_assistant import register_function, register_functions

# Register a single function; this only adds to the existing functions, not replaces them
register_function("function_name", function)

# Register multiple functions; this only adds to the existing functions, not replaces them
register_functions({
    "function_name1": function1,
    "function_name2": function2,
})
```

## 📜 License

This project is licensed under the MIT License.
