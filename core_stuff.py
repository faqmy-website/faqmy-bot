import uuid
from pathlib import Path
import openai
from haystack.document_stores import ElasticsearchDocumentStore
import os

from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor
from trafilatura import fetch_url
from trafilatura import extract
from trafilatura.settings import use_config

STORE_CACHE = {}


def get_document_store(index_name):
    if index_name not in STORE_CACHE:
        STORE_CACHE[index_name] = ElasticsearchDocumentStore(
            host=os.environ.get("ELASTICSEARCH_HOST"),
            username=os.environ.get("ELASTICSEARCH_USER"),
            password=os.environ.get("ELASTICSEARCH_PASSWORD"),
            index=index_name,
            create_index=True,
            similarity="dot_product",
            search_fields=['content']
        )
    return STORE_CACHE[index_name]


# function to preserve only n first words from string that works even for short strings and bigger n
def first_n_words(string, n):
    words = string.split()
    return ' '.join(words[:min(len(words), n)])


def get_nice_response(question, facts):
    retrievedDocsString = '\n'.join(facts)
    # cut knowledge to 100 words
    retrievedDocsString = first_n_words(retrievedDocsString, 200)

    # prompt = "You are a cheery and helpful AI customer-support agent. " \
    #          "Given this context/information:" + \
    #          retrievedDocsString + "\n Respond to this:" + \
    #          question + "\n If you cannot determine the answer from the given information or aren't " \
    #                  "confident in the answer, tell them you don't know and have to check on the answer " \
    #                  "but always tell them something. \nYour response:"

    messages = [
        {"role": "system", "content": "You are a cheery and helpful AI customer-support agent. "
                                      "Given this piece of dialog with another customer: " + retrievedDocsString +
                                      "\nIf you cannot determine the answer from the given information or aren't "
                                      "confident in the answer, tell me you don't know and have to check on "
                                      "the answer but always tell me something"
         },
        # previous messages
        # {"role": "assistant", "content": "Who's there?"},
        # {"role": "user", "content": "Orange."},
        {"role": "user", "content": question},
    ]

    # Set the model
    print('messages=', messages)

    # Call the API
    openai.api_key = os.environ.get("OPENAI_KEY")
    completions = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0,
    )

    generated_text = completions.choices[0]['message']['content']
    output = generated_text.strip()
    print('gpt output=', output)
    return output


def get_random_folder():
    random_folder = str(uuid.uuid4())
    upload_folder = Path("documents/") / random_folder
    upload_folder.mkdir(parents=True, exist_ok=True)
    return upload_folder


def get_url_content(url):
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    downloaded = fetch_url(url)
    output = extract(downloaded, config=config)
    return output


def save_docs_from_folder(target_folder, index):
    docs = convert_files_to_docs(dir_path=target_folder, split_paragraphs=True)
    doc_store = get_document_store(index)

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=50,
        split_respect_sentence_boundary=True,
        split_overlap=0
    )
    small_docs = []
    for document in docs:
        small_docs += processor.process(document)

    print('Amount of docs before splitting = ', len(docs))
    print('Amount of docs after splitting = ', len(small_docs))

    doc_store.write_documents(small_docs)

    print('Success. Document Count after writing:', doc_store.get_document_count())

    resp = [{"id": doc.id, "name": doc.meta['name'], "content": doc.content} for doc in small_docs]
    return resp
