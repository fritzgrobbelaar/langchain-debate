import os
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API keys from secrets.txt
def load_secrets(file_path="secrets.txt"):
    secrets = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            secrets[key] = value.strip("'")
    return secrets

secrets = load_secrets()
os.environ["ANTHROPIC_API_KEY"] = secrets.get("ANTHROPIC_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = secrets.get("DEEPSEEK_API_KEY", "")

# Initialize LLMs
# Agent 1: Claude 3.5 Sonnet (Pro-LangChain)
pro_llm = ChatAnthropic(
    model_name="claude-opus-4-20250514",
    temperature=0.4,  # Focused, technical responses
)

# Agent 2: DeepSeek RAG (Anti-LangChain)
anti_llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.4,
)

# Moderator: Claude 3 Haiku (Smaller Anthropic model)
moderator_llm = ChatAnthropic(
    model_name="claude-3-opus-20240229",
    temperature=0.4,
)

# Define debate context
debate_context = """
Should we allow sperm whale hunting?
"""

# Prompt for Agent 1 (Pro-LangChain, Claude 3.5 Sonnet)
anti_langchain_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}

You want to hunt whales.
Keep answers shorter than 100 words"""
)

# Prompt for Agent 2 (Anti-LangChain, DeepSeek RAG)
pro_langchain_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}
You are a pretentious prick pretending to hate whale hunting.
See if you can get your opponent angry
Keep answers shorter than 100 words.
"""
)

# Moderator prompt (Claude 3 Haiku)
moderator_prompt = PromptTemplate(
    input_variables=["pro_response", "anti_response"],
    template="""
You are a neutral AI migration expert. Review the following arguments from two AI agents debating whale hunting

Pro-LangChain Argument (Claude 3.5 Sonnet):
{pro_response}

Anti-LangChain Argument (DeepSeek RAG):
{anti_response}

Summarize the key points of each position, evaluate the strengths and weaknesses of each argument, and recommend a practical approach for the project, considering automation, time, and maintainability. Keep the response concise and technical.
"""
)

# Create LLM chains using the new RunnableSequence API
pro_chain = pro_langchain_prompt | pro_llm
anti_chain = anti_langchain_prompt | anti_llm
moderator_chain = moderator_prompt | moderator_llm

# Function to run the debate
def run_debate():
    context = debate_context
    for i in range(3):
        print(f"=== Debate Round {i + 1} ===\n")
        print('--total context:', context, '\n\n')
        # Run Agent 2 (Anti-LangChain)
        anti_response = anti_chain.invoke({"context": context})
        print("=== Anti ===")
        print(anti_response.content.replace("\\n", "\n"))
        context = context + "Anti whaler: " + str(anti_response.content) + "\n"
        print("\n")
        
        # Run Agent 1 (Pro-LangChain)
        pro_response = pro_chain.invoke({"context": context})
        print("=== Pro ===")
        print(pro_response.content.replace("\\n", "\n"))
        context = context + "Pro whaler: " + str(pro_response.content) + "\n"
        print("\n")


        # Run Moderator
        moderator_response = moderator_chain.invoke({
            "pro_response": pro_response,
            "anti_response": anti_response
        })
        print("=== Moderator ===")
        print(moderator_response.content.replace("\\n", "\n"))


# Execute the debate
if __name__ == "__main__":
    try:
        run_debate()
    except Exception as e:
        print(f"Error running debate: {e}")