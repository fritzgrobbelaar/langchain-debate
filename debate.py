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
    temperature=0.1,  # Focused, technical responses
)

# Agent 2: DeepSeek RAG (Anti-LangChain)
anti_llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.1,
)

# Moderator: Claude 3 Haiku (Smaller Anthropic model)
moderator_llm = ChatAnthropic(
    model_name="claude-3-opus-20240229",
    temperature=0.8,
)

# Define debate context
debate_context = """
Wie sal wen - 100 mense of 1 Gorilla? Geen wapens, geen voertuie, net 'n geveg in 'n oop veld. Geen klippe, geen bome, net 'n oop veld. Geen ander diere, net mense en 'n gorilla. Geen hulpbronne, net die mense en die gorilla. Geen strategieë, net die mense en die gorilla. Geen voorbereiding, net die mense en die gorilla. Geen onderhandelinge, net die mense en die gorilla. Geen hulp van buite, net die mense en die gorilla. Geen kans om te vlug, net die mense en die gorilla.
"""

# Prompt for Agent 1 (Pro-LangChain, Claude 3.5 Sonnet)
pro_langchain_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}
You are invited to a debate on a fight between 100 people and 1 gorilla.
You argue that 100 people will win.
You respond in Afrikaans
You behave like a 5-year old with many tantrums.
Keep response less than 100 words.
"""
)

# Prompt for Agent 2 (Anti-LangChain, DeepSeek RAG)
anti_langchain_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
{context}
You are invited to a debate on a fight between 100 people and 1 gorilla.
You argue that 1 gorilla will win.
You respond in Afrikaans
You have a very wild imagition and a deep understanding of warfare.
You are very good at thinking outside the box and coming up with creative strategies.
You are technical and analytical.
You are limited to just hand-to-hand combat, no weapons or tools.
Keep response less than 100 words.
""",

)

# Moderator prompt (Claude 3 Haiku)
moderator_prompt = PromptTemplate(
    input_variables=["pro_response", "anti_response"],
    template="""
You are a neutral moderator. Respond in Afrikaans. Review the arguments below:

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
    for i in range(4):
        print(f"=== Debate Round {i + 1} ===\n")
        #print('--total context:', context, '\n\n')
        # Run Agent 2 (Anti-LangChain)
        anti_response = anti_chain.invoke({"context": context})
        print("=== Anti ===")
        print(anti_response.content.replace("\\n", "\n"))
        context = context + "Debater 1: " + str(anti_response.content) + "\n"
        print("\n")
        
        # Run Agent 1 (Pro-LangChain)
        pro_response = pro_chain.invoke({"context": context})
        print("=== Pro ===")
        print(pro_response.content.replace("\\n", "\n"))
        context = context + "Debater 2: " + str(pro_response.content) + "\n"
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