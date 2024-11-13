from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
#1 imports
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

llm= ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

#2 memory objects
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages= True
)

#3
prompt= ChatPromptTemplate(
    input_variables=['content'],
    messages=[
        SystemMessage(content='You are chatbot having a conversation with a human'),
        MessagesPlaceholder(variable_name='chat_history'), #where the memory will be stored
        HumanMessagePromptTemplate.from_template('{content}')
    ]                
)

#4
chain= LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

while True:
    content = input('Your prompt: ')
    if content in ['quit', 'exit', 'bye']:
        print('Goodbye!')
        break

    try: 
        response = chain.invoke({'content': content})
        print(response['text']) 
        print('-' * 50)
    except Exception as e:
        print("An error occurred:", str(e))
        break