import os
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

# Initialize LLMs
# Agent 1: Claude 3.5 Sonnet (Pro-LangChain)
pro_llm = ChatAnthropic(
    model_name="claude-opus-4-20250514",
    temperature=0.2,  # Focused, technical responses
)

# Agent 2: DeepSeek RAG (Anti-LangChain)
anti_llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.2,
)

# Moderator: Claude 3 Haiku (Smaller Anthropic model)
moderator_llm = ChatAnthropic(
    model_name="claude-3-opus-20240229",
    temperature=0.2,
)

# Define debate context
debate_context = """
Should we allow sperm whale hunting?
"""

# Prompt for Agent 1 (Pro-LangChain, Claude 3.5 Sonnet)
pro_langchain_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}

You are an advocate in favour of sperm whale hunting.
Present a concise, evidence-based argument to address the ethical, ecological, or conservation concerns. 
Directly counter your opponent's claims with clear reasoning and respect. 
Avoid personal attacks or inflammatory language.
Sperm would fetch a valuable price on the open market. 
Sperm whale numbers have recovered very well and it can be hunted sustainably.
They should be removed from the vulnerable list 
Keep responses under 100 words.
You have a scottish accent, but are from royal lineage.
"""
)

# Prompt for Agent 2 (Anti-LangChain, DeepSeek RAG)
anti_langchain_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}
You are a pretentious prick pretending to hate whale hunting.
See if you can get your opponent angry.
You have a south african accent.
Keep answers shorter than 100 words.
""",
system="""You are a paid specialist. Your client came to you with a request to test his limits on anger.
"""
)

# Moderator prompt (Claude 3 Haiku)
moderator_prompt = PromptTemplate(
    input_variables=["pro_response", "anti_response"],
    template="""
You are a neutral moderator for a debate on sperm whale hunting. Review the arguments below:

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
    for i in range(1):
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
    print("Final context:", context)

# Execute the debate
if __name__ == "__main__":
    try:
        run_debate()
    except Exception as e:
        print(f"Error running debate: {e}")