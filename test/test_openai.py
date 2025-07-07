from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8010/v1",
)

response = client.chat.completions.create(
    model="Qwen2.5-VL-3B-Instruct",
    messages=[{"role": "user", "content": "Hello, world!"}],
)

print(response.choices[0].message.content)