import asyncio
from inspect_ai.model import get_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

models = [
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


async def test_connection(model_name: str, connection_id: int):
    """
    Test a single connection by generating a response using the specified model.
    """
    model = get_model(f"anthropic/{model_name}")
    try:
        response = await model.generate("Say hello")
        print(f"Connection {connection_id}: {response}")
    except Exception as e:
        print(f"Connection {connection_id} encountered an error: {e}")


async def main():
    max_connections = 20  # Maximum number of concurrent connections
    tasks = []
    model_name = models[0]  # Using the first model for this test
    for i in range(max_connections):
        tasks.append(asyncio.create_task(test_connection(model_name, i + 1)))
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
