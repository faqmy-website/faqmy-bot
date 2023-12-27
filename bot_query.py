import logging
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import BM25Retriever
from haystack.pipelines import DocumentSearchPipeline
import openai
from dotenv import load_dotenv
import os


load_dotenv()


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


document_store = ElasticsearchDocumentStore(
    host=os.environ.get("ELASTICSEARCH_HOST"),
    username=os.environ.get("ELASTICSEARCH_USER"),
    password=os.environ.get("ELASTICSEARCH_PASSWORD"),
    index='one',
    create_index=True,
    similarity="dot_product"
)

retriever = BM25Retriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipeline = ExtractiveQAPipeline(reader, retriever)
# pipeline = DocumentSearchPipeline(retriever)

query = "Do you supply lactose free milk?"
# result = pipeline.run(query, params={"Retriever": {"top_k": 2}})
result = pipeline.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})

print(result)
retrievedDocsString = ''.join([doc.content for doc in result['documents']])



api_key = os.environ.get("OPENAI_KEY")
coreInformation = "Core and additional information about the company that is always good to have on hand."
#companyX = "CompanyX" #Eventually need to add this back in?
prompt = "You are a cheery and helpful customer-support agent.\nGiven this context/information:" + \
         retrievedDocsString + "\n Try your best to respond to this:" + \
         query + "\n If you cannot figure out an answer yourself or from the given information" \
                 ", tell them you don't know and ask for their email address to reach out to them later.\nYour response:"

# Set the model
model = os.environ.get("OPENAI_MODEL")  # Consider using a cheaper model.
openai.api_key = api_key

# Call the API
completions = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=1024,
    temperature=0.1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

generated_text = completions.choices[0].text
print(generated_text)
