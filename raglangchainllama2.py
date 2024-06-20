from torch import cuda, bfloat16, torch

import transformers

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)

model_path = '../Llama-2-7b-chat-hf'
# Define the BitsAndBytes configuration
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Initialize the model configuration from the local folder
model_config = transformers.AutoConfig.from_pretrained(
    model_path
)

# Load the model from the local folder with the specified configuration
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    config=model_config,
    torch_dtype=torch.float16,
)
# Load the tokenizer from the local folder
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids


stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

import os
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader

class LocalTextFileLoader(BaseLoader):
    def __init__(self, directory: str, file_extension: str = ".txt"):
        self.directory = directory
        self.file_extension = file_extension

    def load(self):
        documents = []
        for filename in os.listdir(self.directory):
            if filename.endswith(self.file_extension):
                filepath = os.path.join(self.directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    document = Document(page_content=content, metadata={"source": filepath})
                    documents.append(document)
        return documents

# Example usage:
local_directory = './data'  # Update this path accordingly
loader = LocalTextFileLoader(local_directory)
documents = loader.load()

# Optional: Print the number of documents loaded
print(f"Loaded {len(documents)} documents.")

# Optional: Print the content of the first document to verify
if documents:
    print(f"First document content: {documents[0].page_content[:500]}")  # Print the first 500 characters of the first document


from langchain.text_splitter import RecursiveCharacterTextSplitter

# Increase chunk size to 2000 characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
all_splits = text_splitter.split_documents(documents)


# Print the number of splits
print(f"Total number of splits: {len(all_splits)}")

# Print the first few splits to verify
for i, split in enumerate(all_splits[:5]):  # Change the number 5 to view more or fewer splits
    print(f"Split {i + 1}:")
    print(split.page_content)
    print("\n" + "-"*80 + "\n")

# Optional: Print the metadata of the first few splits to verify
for i, split in enumerate(all_splits[:5]):
    print(f"Metadata for Split {i + 1}: {split.metadata}")


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "../all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)


chat_history = []

query = "when well 6D-11 established?"
result = chain({"question": query, "chat_history": chat_history})

print(result['answer'])
print(result['source_documents'])

chat_history = [(query, result["answer"])]

query = "TELL ME THE PRODUCTION?"
result = chain({"question": query, "chat_history": chat_history})

print(result['answer'])
print(result['source_documents'])
     
