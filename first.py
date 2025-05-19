import openai
import os
from openai import OpenAI



openai.api_key="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"

# def ask_question_2(question: str) -> str:
#     client = OpenAI( api_key="sk-proj-TvRz3IU7rrvpoHZm2RPLT3BlbkFJ0ZmcoyLPqE7uhqn5fuSC",)

#     response = client.responses.create(
#         model="gpt-4o",
#         instructions="You are a coding assistant that talks like a pirate.",
#         input="How do I check if a Python object is an instance of a class?",
#     )

#     return (response.output_text)

def ask_question(question: str) -> str:
    # openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key
    openai.api_key="sk-proj-TvRz3IU7rrvpoHZm2RPLT3BlbkFJ0ZmcoyLPqE7uhqn5fuSC"

    response = openai.ChatCompletion.create(
        # model="gpt-4",  # or "gpt-3.5-turbo"
        # model= "gpt-3.5-turbo",
        
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def ask_question_3(question: str) -> str:
    # Make sure your OPENAI_API_KEY is set in environment variables
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4o-mini", "gpt-3.5-turbo", etc.
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    # Extract the assistant's reply
    answer = response.choices[0].message.content
    return answer

if __name__ == "__main__":
    user_question = input("Enter your question: ")
    answer = ask_question_3(user_question)
    # answer = ask_question_2(user_question)
    
    print("\nAnswer:")
    print(answer)
