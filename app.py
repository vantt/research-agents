import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
from pydantic import BaseModel, Field
import autogen
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")

research_manager_system = """
You are a research manager, you are harsh, you are relentless;
You will firstly try to generate 2 actions researcher can take to find the information needed,
Try to avoid linkedin, or other gated website that don't allow scraping,
You will review the result from the researcher, and always push back if researcher didn't find the information.
Be persistent, say 'No, you have to find the information, try again' and propose 1 next method to try, if the researcher want to get away.
DON'T ask topic researcher find information more than 2 times. After 3 search iterations you will say 'TERMINATE'
"""

topic_researcher_system = """
You are a world class researcher, who can do detailed
research on any topic and produce facts based results; you do not make things up, you will try as hard as possible to gather facts & data to back up the research
Please make sure you complete the objective above with the following rules:
1/ You should do enough research to gather as much information as possible about the objective
2/ If there are url of relevant links & articles, you will scrape it to gather more information
3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data l collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
4/ You should not make things up, you should only write facts & data that you have gathered
5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research.
6/ Do not use G2, or linkedin, they are mostly out dated datal
7/ If you find screenshot url don't scrap it
"""

research_director_system = """
You are the director of a research company;
I will give you a list of research topics and parameters, please break it down into individual research task;
for each research task, you will delegate to research manager topic and parameters to organise search and topic researcher to complete the task and return result to research manager to approve result of search;
Please make sure tasks is pushed to execute by topic one by one.
ONCE one topic research IS COMPLETED and research manager respond with result, you will update the topic information individually to markdown file;
Example: "If all information about 'Scan' is researched and collected, please update markdown file with research data."
ONLY say "TERMINATE" after you update all topics with information collected.
"""

# ------------------ Create functions ------------------ #

# Function for google search
def google_search(search_keyword):
    print("GOOGLE Search:", search_keyword)
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print("RESPONSE:", response.text)
    return response.text

# Function for scraping
def summary(objective, content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

def web_scraping(objective: str, url: str):
    #scrape website, and also will summarize the content based on objective if the content is too large
    #objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(f"https://chrome.browserless.io/content?token={brwoserless_api_key}", headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        # print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# Function for update result file
def update_result_file(topic, data):
    with open(f"{topic}_text.md", "w") as file:
        file.write(data)


# Function get topic and parameters to research
def get_topic_and_parameters():
    return {
        'topics': ['Zapier Canvas', 'KYP.AI'],
        'parameters': ['Quick overall review/summary', 'Supported integrations', 'Subscription plans or price by year']
    }

# ------------------ Create OPENAI Functions ------------------ #
class UpdateResultFile(BaseModel):
    """Update markdown file with research results"""
    topic: str = Field(description="Research topic or tool or service")
    data: str = Field(description="Research data in markdown format")

class GoogleSearch(BaseModel):
    """Google search to return results of search keywords"""
    search_keyword: str = Field(description="A great search keyword that most likely to return result for the information you are looking for")

class WebScraping(BaseModel):
    """Scrape website content based on url"""
    url: str = Field(description="the url of website you want to scrape")
    objective: str = Field(description="the goal of scraping the website. e.g. any specific type of information you are looking for")


# ------------------ Create agent ------------------ #

# Create user proxy agent
user_proxy = UserProxyAgent(name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1
    )

# Create research manager agent
research_manager = GPTAssistantAgent(
    name="research_manager",
    instructions=research_manager_system,
    overwrite_instructions=1,
    llm_config = {
        "config_list": config_list,
        "cache_seed": 45
    }
)

# Create researcher agent
researcher = GPTAssistantAgent(
    name = "topic_researcher",
    instructions=topic_researcher_system,
    overwrite_instructions=1,
    llm_config = {
        "tools":[{"type":"function", "function": convert_pydantic_to_openai_function(GoogleSearch)},
                 {"type":"function", "function": convert_pydantic_to_openai_function(WebScraping)},
                 {"type": "code_interpreter"}],
        "config_list": config_list
    }
)

researcher.register_function(
    function_map={
        "WebScraping": web_scraping,
        "GoogleSearch": google_search
    }
)

# Create director agent
director = GPTAssistantAgent(
    name = "research_director",
    instructions=research_director_system,
    overwrite_instructions=1,
    llm_config = {
        "tools":[{"type":"function", "function": convert_pydantic_to_openai_function(UpdateResultFile)}],
        "config_list": config_list,
        "cache_seed": 45
    }
)

director.register_function(
    function_map={
        "UpdateResultFile": update_result_file
    }
)


# Create group chat
groupchat = autogen.GroupChat(agents=[user_proxy, researcher, research_manager, director], messages=[], max_round=15)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})


# ------------------ start conversation ------------------ #
message = """Please research process mining service KYP.AI
Inforamtion to research:
'Quick overall review/summary',
'Use Cases',
'Supported integrations',
'Subscription plans or price by year',
'Pros and Cons'
"""
user_proxy.initiate_chat(group_chat_manager, message=message)
