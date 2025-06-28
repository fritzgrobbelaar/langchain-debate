import os
import requests
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API keys from secrets.txt
def load_secrets(file_path="../secrets.txt"):
    secrets = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            secrets[key] = value.strip("'")
    return secrets

secrets = load_secrets()
os.environ["ANTHROPIC_API_KEY"] = secrets.get("ANTHROPIC_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = secrets.get("DEEPSEEK_API_KEY", "")

def print_anthropic_models():
    print('available Anthropic models: ', requests.get('https://api.anthropic.com/v1/models',
                                                     headers={
                                                         "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                                         "anthropic-version": "2023-06-01"
                                                     }).json())


# Initialize LLMs
# Agent 1: Claude 3.5 Sonnet (Pro-LangChain)
debator2_llm = ChatAnthropic(
    model_name="claude-3-5-haiku-20241022",
    temperature=0.1,  # Focused, technical responses
)

# Agent 2: DeepSeek RAG (Anti-LangChain)
debator1_llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.1,
)

# Moderator: Claude 3 Haiku (Smaller Anthropic model)
moderator_llm = ChatAnthropic(
    model_name="claude-3-haiku-20240307",
    temperature=0.8,
)

# Define debate context
debate_context = """
School hours in the UK are a topic of ongoing debate. Some argue that the current system, with long school days, is detrimental to student well-being and academic performance. They believe that reducing school hours could lead to better mental health and more effective learning.
On the other hand, others argue that the current school hours are too short, especially for working parents. They contend that a longer school day would provide more structure and support for students, allowing them to engage in extracurricular activities and receive additional academic help.
This debate will explore both sides of the argument, considering the implications for students, parents, and the education system as a whole.
"""

# Prompt for Agent 1 (Pro-LangChain, Claude 3.5 Sonnet)
debator1_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}
You are invited to a debate.
You argue that school hours in the UK are too long and should be reduced to improve student well-being and academic performance.
Keep response less than 100 words.
"""
)

# Prompt for Agent 2 (Anti-LangChain, DeepSeek RAG)
debator2_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}
You are invited to a debate.
You argue that school hours in the UK are too short and the system doesn't prioritize parents working
Keep response less than 100 words.
""",

)

# Moderator prompt (Claude 3 Haiku)
moderator_questions = PromptTemplate(
    input_variables=['context'],
    template="""
You are a neutral moderator trying to focus on the debate. The first question is directed to Debator 1.
Keep response less than 50 words.
{context}
Generate a question based on the context. Direct the question to the debater who is next to respond.

"""
)

# Moderator prompt (Claude 3 Haiku)
moderator_summary = PromptTemplate(
    input_variables=["pro_response", "anti_response"],
    template="""
You are a neutral moderator.  Review the arguments below:

Position 1:
{pro_response}

Position 2:
{anti_response}

Summarize the key points of each position, evaluate the strengths and weaknesses of each argument, and recommend a practical approach for the project, considering automation, time, and maintainability. Keep the response concise and technical.
"""
)

# Create LLM chains using the new RunnableSequence API
debator2_chain = debator2_prompt | debator2_llm
debator1_chain = debator1_prompt | debator1_llm
moderator_questions_chain = moderator_questions | moderator_llm
moderator_chain = moderator_summary | moderator_llm

# Function to run the debate
def run_debate():
    context = debate_context
    for i in range(4):
        print(f"=== Debate Round {i + 1} ===\n")
        #print('--total context:', context, '\n\n')
        # Run Agent 2 (Anti-LangChain)
        moderator_response = moderator_questions_chain.invoke({"context": context})
        print("=== Moderator Question to Debater 1 ===")
        print(moderator_response.content.replace("\\n", "\n"))
        context = context + "\nModerator: " + str(moderator_response.content) + "\n"


        debator1_response = debator1_chain.invoke({"context": context})
        print("=== 1 ===")
        print(debator1_response.content.replace("\\n", "\n"))
        context = context + "Debater 1: " + str(debator1_response.content) + "\n"
        print("\n")
        
        moderator_response = moderator_questions_chain.invoke({"context": context})
        print("=== Moderator Question to Debater 2 ===")
        print(moderator_response.content.replace("\\n", "\n"))
        context = context + "Moderator: " + str(moderator_response.content) + "\n"

        # Run Agent 1 (Pro-LangChain)
        debator2_response = debator2_chain.invoke({"context": context})
        print("=== 2 ===")
        print(debator2_response.content.replace("\\n", "\n"))
        context = context + "Debater 2: " + str(debator2_response.content) + "\n"
        print("\n")


        # Run Moderator
    moderator_response = moderator_chain.invoke({
        "pro_response": debator1_response,
        "anti_response": debator2_response
    })
    print("\n\n------Final context:", context)
    print("=== Moderator ===")
    print(moderator_response.content.replace("\\n", "\n"))
    

# Execute the debate
if __name__ == "__main__":
    try:
        run_debate()
    except Exception as e:
        print(f"Error running debate: {e}")