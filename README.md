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

To use this package, you need to initialize it with your OpenAI client.  
Here's an example using AsyncOpenAI:

```python
import asyncio
from openai_assistant import init, OpenAIAssistant
from openai import AsyncOpenAI

# Initialize the OpenAI client with API key
openai = AsyncOpenAI(api_key="sk-123456")

# Initialize OpenAI assistants
assistants = init(openai)

# Define your assistant or agent class
class TestAgent(OpenAIAssistant):
    def __init__(self, assistant_id, thread_id=None, callback=None):
        # or just put your assistant id here
        assistant_id = "your-assistant-id"
        super().__init__(assistant_id, thread_id, callback)

    @classmethod
    async def create(cls, assistant_id, thread_id=None, callback=None):
        # You can remove assistant_id from arguments, if you put it in __init__
        self = cls(assistant_id, thread_id, callback)
        await self.initialize_thread_id()
        return self

    async def submit_request(self, user_input: str):
        return await self.send_request(user_input)
    
    async def send_request_image_base64(self, user_input: str, base64_images: list):
        return await self.send_request_image_base64(user_input, base64_images)
        
    async def send_request_image_url(self, user_input: str, urls: list):
        return await self.send_request_image_url(user_input, urls)

# Create an agent instance and submit a request
async def main():
    agent = await TestAgent.create()
    response = await agent.submit_request("Hello, how are you?")
    print(response)

    # Base64 image example
    response = await agent.send_request_image_base64("What is in this image?", ["data:image/jpeg;base64,/9jS..."])
    print(response)

    # URL image example
    response = await agent.send_request_image_url("What is in this image?", ["https://example.com/image.jpg"])
    print(response)

# Run the main function
asyncio.run(main())
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
