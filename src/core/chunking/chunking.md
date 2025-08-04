Chunking techniques with Langchain and LlamaIndex
In our last blog, we talked about chunking and why it is necessary for processing data through LLMs.

By Prashant Kumar
14 min. readView original



In our last blog, we talked about chunking and why it is necessary for processing data through LLMs. We covered some simple techniques to perform text chunking.


In this blog, we will comprehensively cover all the chunking techniques available in Langchain and LlamaIndex.



The aim is to get the data in a format where it can be used for anticipated tasks, and retrieved for value later. Rather than asking “How should I chunk my data?”, the actual question should be “What is the optimal way for me to pass data to my language model that it needs for its task?”


Let's begin with Langchain first 


Chunking Techniques in Langchain



Before jumping into chunking, make sure to first install Langchain-text-splitters


! pip install langchain-text-splitters


These snippets only cover the relevant sections of code. To follow along with the working code, please use the following google colab:


Google Colab



Text Character Splitting


This is the simplest method. This splits based on characters and measures chunk length by number of characters.


from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0].page_content)


This outputs chunks with 1000 characters:


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:

Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. For 220 years, our leaders have fulfilled this duty. They have done so during periods of prosperity and tranquility. And they have done so in the midst of war and depression; at moments of great strife and great struggle.


Recursive Character Splitting


This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""] . It includes overlapping text which helps build context between text splits.


# Recursive Split Character

# This is a long document we can split up.
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([state_of_the_union])
print("Chunk 2: ", texts[1].page_content)
print("Chunk 3: ", texts[2].page_content)


Here are Chunk 2 and Chunk 3 in output showing an overlap of 100 character:


Chunk 2:  It's tempting to look back on these moments and assume that our progress was inevitable, that America was always destined to succeed. But when the Union was turned back at Bull Run and the Allies first landed at Omaha Beach, victory was very much in doubt. When the market crashed on Black Tuesday and civil rights marchers were beaten on Bloody Sunday, the future was anything but certain. These were times that tested the courage of our convictions and the strength of our union. And despite all our divisions and disagreements, our hesitations and our fears, America prevailed because we chose to move forward as one nation and one people.

Again, we are tested. And again, we must answer history's call.

Chunk 3:  Again, we are tested. And again, we must answer history's call.
One year ago, I took office amid two wars, an economy rocked by severe recession, a financial system on the verge of collapse and a government deeply in debt. Experts from across the political spectrum warned that if we did not act, we might face a second depression. So we acted immediately and aggressively. And one year later, the worst of the storm has passed.
But the devastation remains. One in 10 Americans still cannot find work. Many businesses have shuttered. Home values have declined. Small towns and rural communities have been hit especially hard. For those who had already known poverty, life has become that much harder.
This recession has also compounded the burdens that America's families have been dealing with for decades -- the burden of working harder and longer for less, of being unable to save enough to retire or help kids with college.


HTML Section Splitter


HTML Section Splitter is a “structure-aware” chunker that splits text at the element level and adds metadata for each header “relevant” to any given chunk.


url = "https://www.utoronto.ca/"

# Send a GET request to the URL
response = requests.get(url)
if response.status_code == 200:
    html_doc = response.text

headers_to_split_on = [
    ("h1", "Header 1"),
    ("p", "paragraph")
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_doc)
html_header_splits[0].page_content


This outputs the HTML Header element:


Welcome to University of Toronto


Code Splitter


CodeTextSplitter allows you to split your code with multiple languages supported. Supported Languages are Python, JS, TS, C, Markdown, Latex, HTML, and Solidity.


from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

with open("code.py") as f:
    code = f.read()

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=100, chunk_overlap=0
)
python_docs = python_splitter.create_documents([code])
python_docs[0].page_content


This splits code according to the Python language in chunks of 100 characters:


from youtube_podcast_download import podcast_audio_retreival


Recursive Json Splitting


This JSON splitter goes through JSON data from the deepest levels first and creates smaller JSON pieces. It tries to keep nested JSON objects intact but may divide them if necessary to ensure that the chunks fall within a specified range from the minimum to the maximum chunk size.


from langchain_text_splitters import RecursiveJsonSplitter
import json
import requests

json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = splitter.split_json(json_data=json_data)
json_chunks[0]


Result:


{'openapi': '3.1.0',
 'info': {'title': 'LangSmith', 'version': '0.1.0'},
 'servers': [{'url': 'https://api.smith.langchain.com',
   'description': 'LangSmith API endpoint.'}]}


Semantic Splitting


This method splits the text based on semantic similarity. Here we’ll use OpenAI Embeddings for semantic similarity. 


#!pip install --quiet langchain_experimental langchain_openai

import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Add OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "sk-****"

with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = SemanticChunker(OpenAIEmbeddings())

docs = text_splitter.create_documents([state_of_the_union])
print(docs[0].page_content)


Semantic splitting result: 


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:

Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. For 220 years, our leaders have fulfilled this duty.


Split by Tokens


Language models come with a token limit, which you can not surpass.To prevent issues, make sure to track the token count when dividing your text into chunks. Be sure to use the same tokenizer as the language model to count the tokens in your text.


For now, we’ll use Tiktoken as a tokenizer


# ! pip install --upgrade --quiet tiktoken
from langchain_text_splitters import CharacterTextSplitter

with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

print(texts[0])


Token Splitting using Tiktoken results:


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:

Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. For 220 years, our leaders have fulfilled this duty. They have done so during periods of prosperity and tranquility. And they have done so in the midst of war and depression; at moments of great strife and great struggle.


These are the most important text-splitting/ chunking techniques using Langchain. Now let’s see similar techniques with their implemented in LlamaIndex.


LlamaIndex Chunking Techniques with Implementation



Make sure to install llama_index package.


! pip install llama_index tree_sitter tree_sitter_languages -q


In LlamaIndex, Node parsers terminology is used which breaks down a list of documents into Node objects where each node represents a distinct chunk of the parent document, inheriting all attributes from the Parent document to the Children nodes.


Node Parser — Simple File


To make it easier to read nodes, there are different file-based parsers you can use. They’re designed for different kinds of content, like JSON or Markdown. One simple way is to use the FlatFileReader with the SimpleFileNodeParser. This setup automatically picks the right parser for the content type you’re dealing with.


from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

# Download for running any text file
!wget https://raw.githubusercontent.com/lancedb/vectordb-recipes/main/README.md

md_docs = FlatReader().load_data(Path("README.md"))

parser = SimpleFileNodeParser()

# Additionally, you can augment this with a text-based parser to accurately 
# handle text length

md_nodes = parser.get_nodes_from_documents(md_docs)
md_nodes[0].text


This outputs:


VectorDB-recipes
<br />
Dive into building GenAI applications!
This repository contains examples, applications, starter code, & tutorials to help you kickstart your GenAI projects.

- These are built using LanceDB, a free, open-source, serverless vectorDB that **requires no setup**. 
- It **integrates into python data ecosystem** so you can simply start using these in your existing data pipelines in pandas, arrow, pydantic etc.
- LanceDB has **native Typescript SDK** using which you can **run vector search** in serverless functions!

<img src="https://github.com/lancedb/vectordb-recipes/assets/5846846/d284accb-24b9-4404-8605-56483160e579" height="85%" width="85%" />

<br />
Join our community for support - <a href="https://discord.gg/zMM32dvNtd">Discord</a> •
<a href="https://twitter.com/lancedb">Twitter</a>

---

This repository is divided into 3 sections:
- [Examples](#examples) - Get right into the code with minimal introduction, aimed at getting you from an idea to PoC within minutes!
- [Applications](#projects--applications) - Ready to use Python and web apps using applied LLMs, VectorDB and GenAI tools
- [Tutorials](#tutorials) - A curated list of tutorials, blogs, Colabs and courses to get you started with GenAI in greater depth.


Node Parser — HTML


This node parser uses Beautiful Soup to understand raw HTML content. It’s set up to read certain HTML tags automatically, like “p”, and “h1” through “h6”, “li”, “b”, “i”, “u”, and “section”. You can also choose which tags it pays attention to if you want to customize it.


import requests
from llama_index.core import Document
from llama_index.core.node_parser import HTMLNodeParser

# URL of the website to fetch HTML from
url = "https://www.utoronto.ca/"

# Send a GET request to the URL
response = requests.get(url)
print(response)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the HTML content from the response
    html_doc = response.text
    document = Document(id_=url, text=html_doc)

    parser = HTMLNodeParser(tags=["p", "h1"])
    nodes = parser.get_nodes_from_documents([document])
    print(nodes)
else:
    # Print an error message if the request was unsuccessful
    print("Failed to fetch HTML content:", response.status_code)


This returns the output with an HTML tag in the metadata:


[TextNode(id_='bf308ea9-b937-4746-8645-c8023e2087d7', embedding=None, metadata={'tag': 'h1'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='https://www.utoronto.ca/', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='247fb639a05bc6898fd1750072eceb47511d3b8dae80999f9438e50a1faeb4b2'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c280bdf-7373-4be8-8e70-6360848581e9', node_type=<ObjectType.TEXT: '1'>, metadata={'tag': 'p'}, hash='3e989bb32b04814d486ed9edeefb1b0ce580ba7fc8c375f64473ddd95ca3e824')}, text='Welcome to University of Toronto', start_char_idx=2784, end_char_idx=2816, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), TextNode(id_='7c280bdf-7373-4be8-8e70-6360848581e9', embedding=None, metadata={'tag': 'p'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='https://www.utoronto.ca/', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='247fb639a05bc6898fd1750072eceb47511d3b8dae80999f9438e50a1faeb4b2'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='bf308ea9-b937-4746-8645-c8023e2087d7', node_type=<ObjectType.TEXT: '1'>, metadata={'tag': 'h1'}, hash='e1e6af749b6a40a4055c80ca6b821ed841f1d20972e878ca1881e508e4446c26')}, text='In photos: Under cloudy skies, U of T community gathers to experience near-total solar eclipse\nYour guide to the U of T community\nThe University of Toronto is home to some of the world’s top faculty, students, alumni and staff. U of T Celebrates recognizes their award-winning accomplishments.\nDavid Dyzenhaus recognized with Gold Medal from Social Sciences and Humanities Research Council\nOur latest issue is all about feeling good: the only diet you really need to know about, the science behind cold plunges, a uniquely modern way to quit smoking, the “sex, drugs and rock ‘n’ roll” of university classes, how to become a better workplace leader, and more.\nFaculty and Staff\nHis course about the body is a workout for the mind\nProfessor Doug Richards teaches his students the secret to living a longer – and healthier – life\n\nStatement of Land Acknowledgement\nWe wish to acknowledge this land on which the University of Toronto operates. For thousands of years it has been the traditional land of the Huron-Wendat, the Seneca, and the Mississaugas of the Credit. Today, this meeting place is still the home to many Indigenous people from across Turtle Island and we are grateful to have the opportunity to work on this land.\nRead about U of T’s Statement of Land Acknowledgement.\nUNIVERSITY OF TORONTO - SINCE 1827', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n')]


Node Parser — JSON


To handle JSON documents, we’ll use a JSON parser.


from llama_index.core.node_parser import JSONNodeParser

url = "https://housesigma.com/bkv2/api/search/address_v2/suggest"

payload = {"lang": "en_US", "province": "ON", "search_term": "Mississauga, ontario"}

headers = {
    'Authorization': 'Bearer 20240127frk5hls1ba07nsb8idfdg577qa'
}

response = requests.post(url, headers=headers, data=payload)

if response.status_code == 200:
    document = Document(id_=url, text=response.text)
    parser = JSONNodeParser()

    nodes = parser.get_nodes_from_documents([document])
    print(nodes[0])
else:
    print("Failed to fetch JSON content:", response.status_code)


Above code outputs:


status True data house_list id_listing owJKR7PNnP9YXeLP data
house_list house_type_in_map D data house_list price_abbr 0.75M data
house_list price 749,000 data house_list price_sold 690,000 data
house_list tags Sold data house_list list_status public 1 data
house_list list_status live 0 data house_list list_status s_r Sale


Node Parser — Markdown


To handle Markdown files, we’ll use a Markdown parser.


# Markdown
from llama_index.core.node_parser import MarkdownNodeParser

md_docs = FlatReader().load_data(Path("README.md"))
parser = MarkdownNodeParser()

nodes = parser.get_nodes_from_documents(md_docs)
nodes[0].text


This output same as the Simple File parser showed:


Now we have seen the node parser, let's see how to do chunking by utilizing these node parsers.


VectorDB-recipes
<br />
Dive into building GenAI applications!
This repository contains examples, applications, starter code, & tutorials to help you kickstart your GenAI projects.

- These are built using LanceDB, a free, open-source, serverless vectorDB that **requires no setup**. 
- It **integrates into python data ecosystem** so you can simply start using these in your existing data pipelines in pandas, arrow, pydantic etc.
- LanceDB has **native Typescript SDK** using which you can **run vector search** in serverless functions!

<img src="https://github.com/lancedb/vectordb-recipes/assets/5846846/d284accb-24b9-4404-8605-56483160e579" height="85%" width="85%" />

<br />
Join our community for support - <a href="https://discord.gg/zMM32dvNtd">Discord</a> •
<a href="https://twitter.com/lancedb">Twitter</a>

---

This repository is divided into 3 sections:
- [Examples](#examples) - Get right into the code with minimal introduction, aimed at getting you from an idea to PoC within minutes!
- [Applications](#projects--applications) - Ready to use Python and web apps using applied LLMs, VectorDB and GenAI tools
- [Tutorials](#tutorials) - A curated list of tutorials, blogs, Colabs and courses to get you started with GenAI in greater depth.


Code Splitter


Code Splitter allows you to split your code with multiple languages supported. You can just mention the name of the language and do splitting.


# Code Splitting

from llama_index.core.node_parser import CodeSplitter
documents = FlatReader().load_data(Path("app.py"))
splitter = CodeSplitter(
    language="python",
    chunk_lines=40,  # lines per chunk
    chunk_lines_overlap=15,  # lines overlap between chunks
    max_chars=1500,  # max chars per chunk
)
nodes = splitter.get_nodes_from_documents(documents)
nodes[0].text


This creates a chunk of 40 lines of code as a result.


Sentence Splitting


The SentenceSplitter attempts to split text while respecting the boundaries of sentences.


from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)
nodes[0].text


This results in a chunk of 1024-size:


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:
Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. For 220 years, our leaders have fulfilled this duty. They have done so during periods of prosperity and tranquility. And they have done so in the midst of war and depression; at moments of great strife and great struggle.
It's tempting to look back on these moments and assume that our progress was inevitable, that America was always destined to succeed. But when the Union was turned back at Bull Run and the Allies first landed at Omaha Beach, victory was very much in doubt. When the market crashed on Black Tuesday and civil rights marchers were beaten on Bloody Sunday, the future was anything but certain. These were times that tested the courage of our convictions and the strength of our union. And despite all our divisions and disagreements, our hesitations and our fears, America prevailed because we chose to move forward as one nation and one people.
Again, we are tested. And again, we must answer history's call.


Sentence Window Node Parser


The SentenceWindowNodeParser functions similarly to other node parsers, but with the distinction of splitting all documents into individual sentences. Each resulting node also includes the neighboring “window” of sentences surrounding it in the metadata. It’s important to note that this metadata won’t be accessible to the LLM or embedding model.


import nltk
from llama_index.core.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_sentence",
)
nodes = node_parser.get_nodes_from_documents(documents)
nodes[0].text


It results:


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:
Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. 


Semantic Splitting 


Semantic chunking offers a new method in which, instead of breaking text into chunks of a fixed size, a semantic splitter dynamically chooses where to split between sentences, based on embedding similarity.


from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import os

# Add OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "sk-****"

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

nodes = splitter.get_nodes_from_documents(documents)
nodes[0].text


Semantic Splitting results:


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:

Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. For 220 years, our leaders have fulfilled this duty. 


Token Text Splitting


The TokenTextSplitter attempts to split to a consistent chunk size according to raw token counts.


from llama_index.core.node_parser import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=254,
    chunk_overlap=20,
    separator=" ",
)
nodes = splitter.get_nodes_from_documents(documents)
nodes[0].text


Token Splitting results:


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:
Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. For 220 years, our leaders have fulfilled this duty. They have done so during periods of prosperity and tranquility. And they have done so in the midst of war and depression; at moments of great strife and great struggle.
It's tempting to look back on these moments and assume that our progress was inevitable, that America was always destined to succeed. But when the Union was turned back at Bull Run and the Allies first landed at Omaha Beach, victory was very much in doubt. When the market crashed on Black Tuesday and civil rights marchers were beaten on Bloody Sunday, the future was anything but certain. These were times that tested the courage of our convictions and the strength of our union. And despite all our divisions and disagreements, our hesitations and our fears, America prevailed because we chose to move forward as one nation and one people.
Again, we are tested. And again, we must answer history's call.
One year ago, I took office amid two wars, an economy


Hierarchical Node Parser


This node parser divides nodes into hierarchical structures, resulting in multiple hierarchies of various chunk sizes from a single input. Each node includes a reference to its parent node.


from llama_index.core.node_parser import HierarchicalNodeParser

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 254, 128]
)

nodes = node_parser.get_nodes_from_documents(documents)
nodes[0].text


The results of the Hierarchical parser look like:


Madame Speaker, Vice President Biden, members of Congress, distinguished guests, and fellow Americans:
Our Constitution declares that from time to time, the president shall give to Congress information about the state of our union. For 220 years, our leaders have fulfilled this duty. They have done so during periods of prosperity and tranquility. And they have done so in the midst of war and depression; at moments of great strife and great struggle.
It's tempting to look back on these moments and assume that our progress was inevitable, that America was always destined to succeed. But when the Union was turned back at Bull Run and the Allies first landed at Omaha Beach, victory was very much in doubt. When the market crashed on Black Tuesday and civil rights marchers were beaten on Bloody Sunday, the future was anything but certain. These were times that tested the courage of our convictions and the strength of our union. And despite all our divisions and disagreements, our hesitations and our fears, America prevailed because we chose to move forward as one nation and one people.
Again, we are tested. And again, we must answer history's call.
One year ago, I took office amid two wars, an economy rocked by severe recession, a financial system on the verge of collapse and a government deeply in debt. Experts from across the political spectrum warned that if we did not act, we might face a second depression. So we acted immediately and aggressively. And one year later, the worst of the storm has passed.
But the devastation remains. One in 10 Americans still cannot find work. Many businesses have shuttered. Home values have declined. Small towns and rural communities have been hit especially hard. For those who had already known poverty, life has become that much harder.
This recession has also compounded the burdens that America's families have been dealing with for decades -- the burden of working harder and longer for less, of being unable to save enough to retire or help kids with college.
So I know the anxieties that are out there right now. They're not new. These struggles are the reason I ran for president. These struggles are what I've witnessed for years in places like Elkhart, Ind., and Galesburg, Ill. I hear about them in the letters that I read each night.


Colab Walkthrough


Google Colab



Conclusion 


Langchain and Llama Index are popular tools, and one of the key things they do is "chunking." This means breaking down data into smaller pieces, which is important for making the tools work well. These platforms provide a variety of ways to do chunking, creating a unified solution for processing data efficiently. This article will guide you through all the chunking techniques you can find in Langchain and Llama Index.


References
	•	The aim is to get the data in a format where it can be used for anticipated tasks, and retrieved for value later. Rather than asking “How should I chunk my data?”, the actual question should be “What is the optimal way for me to pass data to my language model that it needs for its task?” - link
Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking
Hai-Toan Nguyen
Corresponding author
Tien-Dat Nguyen

Viet-Ha Nguyen
Abstract
Retrieval-Augmented Generation (RAG) systems commonly use chunking strategies for retrieval, which enhance large language models (LLMs) by enabling them to access external knowledge, ensuring that the retrieved information is up-to-date and domain-specific. However, traditional methods often fail to create chunks that capture sufficient semantic meaning, as they do not account for the underlying textual structure. This paper proposes a novel framework that enhances RAG by integrating hierarchical text segmentation and clustering to generate more meaningful and semantically coherent chunks. During inference, the framework retrieves information by leveraging both segment-level and cluster-level vector representations, thereby increasing the likelihood of retrieving more precise and contextually relevant information. Evaluations on the NarrativeQA, QuALITY, and QASPER datasets indicate that the proposed method achieved improved results compared to traditional chunking techniques.
Keywords: Retrieval Augmented Generation Semantic Chunking Text Segmentation.
1Introduction
In the field of artificial intelligence (AI), processing unstructured data has become essential. Large Language Models (LLMs), such as OpenAI’s GPT1
1
https://openai.com/
, is capable of performing complex tasks and supporting a wide range of applications [17, 18]. While these models are effective at generating responses and automating tasks, their performance is often limited by the quality of the data they process.
Updating these models via fine-tuning or other modifications can be challenging, particularly when dealing with large text corpora [9, 16]. One common approach to address this issue involves dividing large volumes of text into smaller, manageable chunks. This method is widely used in question-answering systems, where splitting texts into smaller units improves retrieval accuracy [4]. This retrieval-based method, known as Retrieval-Augmented Generation (RAG) [16], enhances LLMs by allowing them to reference external knowledge, making it easier to ensure that the retrieved information is up-to-date and domain-specific. However, current retrieval-augmented methods have limitations, particularly in the chunking process. Traditional chunking approaches often fail to capture sufficient semantic meaning as they do not account for the underlying textual structure. This limitation becomes especially problematic for answering complex queries that require understanding multiple parts of a document, such as books in NarrativeQA [13] or research papers in QASPER [5].
To address these challenges, we propose a novel framework that improves the retrieval process by generating chunks that either capture local context (segments) or clusters of related segments that represent higher-level semantic coherence. The key components of this framework are:
	1	1. Text Segmentation: A supervised text segmentation model is applied to divide the document into smaller, coherent segments. This ensures that each segment preserves meaningful local context and is not cut off inappropriately.
	2	2. Chunk Clustering: After segmentation, unsupervised clustering combines related segments based on semantic similarity and their relative positions. This process creates clusters that maintain sequential structure and capture broader semantic relationships between segments.
	3	3. Multiple-Vector Based Retrieval: Each text chunk is represented by multiple vectors: several for individual segments within the chunk, and one for the cluster itself. This approach provides more options for matching during retrieval, as having multiple vectors to compare against increases the likelihood of finding a more precise match, whether based on specific segment details or broader cluster context.
2Related Work
RAG enables LLMs to access external knowledge during inference, improving their performance on domain-specific tasks. Research by Gao et al. [7] demonstrates that RAG enhances various Natural Language Processing (NLP) tasks, particularly in improving factual accuracy. Consequently, several studies have focused on to enhance RAG’s performance, either by refining the prompts used for generation [22] or by incorporating diverse retrieval strategies [11].
A key factor in optimizing RAG systems is how documents are chunked for retrieval. Various traditional chunking methods have been developed, with open-source frameworks like LangChain2
2
https://www.langchain.com/
 and LlamaIndex3
3
https://docs.llamaindex.ai/en/stable/
 offering techniques for splitting and filtering documents. Fixed-size chunking [20], which divides text into equal-length segments, is straightforward to implement but often fails to capture the semantic or structural nuances of the text. Recursive splitting [15], which segments text based on markers like newlines or spaces, can be effective when documents have clear formatting. Semantic chunking [12], which groups sentences based on cosine similarity in embedding space, produces more coherent chunks, but it can sometimes lead to inconsistent boundaries or miss broader context by focusing too narrowly on sentence-level similarity.
Recent research has made strides in improving chunk retrieval and ensuring more contextually relevant chunks in RAG systems. LongRAG [10] proposes using longer retrieval units, such as entire Wikipedia documents, to reduce the overall size of the corpus. In contrast, RAPTOR [23] creates multi-level chunk hierarchies, progressing from detailed content at the leaf nodes to more abstract summaries at higher levels, thereby maintaining relevance while enabling the model to handle both granular and high-level content.
Another area of improvement in RAG is the integration of Knowledge Graphs (KGs) [9]. KGs enhance retrieval accuracy by linking related entities and concepts. For instance, GraphRAG [6] uses entity extraction and query-focused summarization to organize chunks. However, this can disrupt the natural flow of the text by grouping chunks from different sections. In contrast, our approach prioritizes cohesive chunks that maintain the original text structure. By using unsupervised clustering based on text segmentation, our chunks preserve both semantic unity and sequential order, leading to a more coherent retrieval process.
Chunking and text segmentation share similar goals, as both methods aim to divide text into manageable, coherent units. Early text segmentation approaches, such as TextTiling [8] and TopicTiling [21], used unsupervised methods to detect shifts in lexical cohesion. With advancements in neural networks, supervised models have emerged, such as SECTOR [1] and S-LSTM [2], which leverage Long Short-Term Memory (LSTM) or bidirectional LSTMs to predict segment boundaries from labeled data. However, despite these advancements, text segmentation has traditionally been treated as a standalone task. Our method integrates text segmentation to improve document chunking, leading to more accurate and coherent retrieval.
3Hierarchical Text Segmentation Framework for RAG

Figure 1:Overview of our framework, incorporating text segmentation during the indexing process.
3.1Overview
Fig. 1 illustrates our proposed RAG framework, which introduces a hierarchical segmentation and clustering pipeline to enhance chunking accuracy and retrieval relevance. During the indexing phase, each document 
D
 is segmented into coherent segments 
S
i
, and related segments are grouped into clusters 
C
j
. Both segment and cluster embeddings are then computed and stored, as represented by the following equation:

Indexing
⁢
(
D
)
=
{
(
E
s
,
E
c
)
∣
E
s
=
f
segment
⁢
(
S
i
)
,
E
c
=
f
cluster
⁢
(
C
j
)
}

(1)
In the retrieval phase, for each chunk 
C
i
, the system computes the cosine similarity between the query 
q
 and all embeddings associated with the chunk, including both segment embeddings 
E
s
 and cluster embeddings 
E
c
. The most relevant embeddings are selected based on similarity scores, calculated as:

cos
⁢
(
q
,
C
i
)
=
max
⁡
(
cos
⁢
(
q
,
E
s
⁢
1
)
,
…
,
cos
⁢
(
q
,
E
s
⁢
m
)
,
cos
⁢
(
q
,
E
c
)
)

(2)
where 
E
s
⁢
1
,
E
s
⁢
2
,
…
,
E
s
⁢
m
 are segment embeddings for chunk 
C
i
, and 
E
c
 is the cluster embedding for chunk 
C
i
. The system ranks the similarity scores and selects the top-
k
 chunks for processing by the LLMs to generate the response.

Figure 2:Illustration of the framework’s chunking strategy: The text is first segmented into coherent parts using a text segmentation model. These segments are then clustered based on semantic similarities and their sequential order.
3.2Text Segmentation and Clustering: A Bottom-Up Approach
In theory, a hierarchical structure would naturally suit a top-down segmentation approach, where the document is first divided into broader sections and then broken down into smaller units. However, due to the limitations of current text segmentation models—such as the lack of multi-level training data and difficulties with processing long documents—we propose a bottom-up approach.
This bottom-up approach starts by using supervised methods to identify smaller, cohesive segments, which are then grouped into larger, meaningful units through unsupervised clustering techniques. While top-down segmentation may seem intuitive, especially for capturing hierarchical relationships, current models struggle with the complexity of longer texts like books or research articles. Fig. 2 shows the structure of this approach. The bottom-up approach works well with RAG’s retrieval mechanism, which does not depend on a strict sequential structure. In RAG, chunks are retrieved based on their relevance to the query, allowing more flexibility in building document representation from smaller units.
3.2.1Text Segmentation
The model used in our research, introduced by Koshorek et al. [14], is a neural network designed for supervised text segmentation. It predicts whether a sentence marks the end of a section by using a bidirectional LSTM to process sentence embeddings. These embeddings are generated in a previous layer, where another bidirectional LSTM processes the words in each sentence and applies max-pooling to produce fixed-length representations. The model is trained to label each sentence as either a ’1’ (indicating the end of a section) or a ’0’ (indicating continuation), by learning segmentation patterns from the training data. Fig. 3 demonstrated an example of how a document is segmented.

Figure 3:A biography of Alexander Hamilton is being predicted by a text segmentation model. The model predicts segment boundaries by labeling each sentence with a 1 or a 0, where 1 marks the end of a segment and 0 otherwise.
3.2.2Clustering
We adapted the clustering method from Glavias et al. [6], where instead of clustering sentences into segments, we grouped segments into cohesive clusters. The process is outlined as follows:
	1	1. Graph Construction: After text segmentation, each segment is represented as a node in a relatedness graph G=(V,E), where V consists of the segments. An edge is added between two segments Si and Sj if their similarity exceeds a predefined threshold. The threshold τ is set as:
	2	
	3	τ=μ+k⋅σ
	4	
	5	(3)
	6	where μ is the mean similarity between all segments, σ is the standard deviation, and k is a parameter that controls the sensitivity of the connections between segments.
	7	2. Maximal Clique Detection: The task is to identify all maximal cliques in the graph G, which are then stored in a set Q.
	8	3. Initial Clustering: An initial set of clusters is created by merging adjacent segments that are part of at least one clique Q∈Q in graph G
	9	4. Merge Clusters: Adjacent clusters ci and ci+1 are merged if there is at least one clique Q∈Q containing at least one segment from ci and one from ci+1. Table 1 provides an illustration of this merging process.
	10	5. Final Merging: Any remaining single-sentence clusters are merged with the nearest neighboring cluster, based on cosine similarity, to ensure no isolated segments remain.
	11	6. Cluster Embedding: Once clusters are finalized, embeddings for each cluster are calculated by applying mean pooling over the vector representations of the segments within the cluster.
Table 1:Example of merging segments into clusters from cliques.
Step
Sets
Cliques 
Q
 
{1, 2, 6}, {2, 4, 7}, {3, 4, 5}, {1, 6, 7}
Init. clus.
{1, 2}, {3, 4, 5}, {6, 7}
Merge clus.
{1, 2, 3, 4, 5}, {6, 7}
4Experiments
4.1Datasets
Our experiments were conducted on three datasets: NarrativeQA, QuALITY, and QASPER.
NarrativeQA contains 1,572 documents, including books and movie transcripts [13]. The task requires a comprehensive understanding of the entire narrative to answer questions accurately, testing the framework’s ability to comprehend and process longer, complex texts. We assess performance using ROUGE-L, BLEU-1, BLEU-4, and METEOR metrics.
The QuALITY dataset [19] consists of multiple-choice questions, each accompanied by a context passage in English. This dataset is particularly useful for testing the retrieval effectiveness of the system, as answering these questions often requires reasoning across an entire document. Accuracy is used as the evaluation metric for this dataset.
Lastly, QASPER is a benchmark dataset designed for question-answering tasks in scientific NLP papers [5], where the answers are embedded within full-text documents, rather than being located in a specific section or abstract. Performance is measured using the F1 score.
4.2Experimentation Settings
For our experiments, we used the GPT-4o-mini4
4
https://platform.openai.com/docs/models/gpt-4o-mini
 model as the reader and the BAAI/bge-m35
5
https://huggingface.co/BAAI/bge-m3
 model for embedding generation. The embeddings were stored and retrieved using FAISS6
6
https://github.com/facebookresearch/faiss
, an efficient vector database for similarity search.
We evaluated our chunking strategies by testing different average chunk sizes: 512, 1024, and 2048 tokens. Two retrieval methods were tested: one combining segment and cluster vector search, and one using cluster vector search alone. These strategies were compared against fixed-size chunking baselines of 256, 512, 1024, and 2048 tokens, as well as the semantic chunking method by Greg Kamradt [12], which uses an average embedding size of 256 tokens.
To maintain consistent input size for the LLM across different chunking strategies, we retrieved a proportional number of chunks based on the segment length. Specifically, we retrieved 20 chunks for 256-token chunks, 8 chunks for 512-token chunks, 4 chunks for 1024-token chunks, and 2 chunks for 2048-token chunks. This approach ensured that the total number of tokens retrieved approximately 4096, allowing for a fair comparison across all chunking strategies.
4.3Chunking Setup
Text Segmentation As mentioned earlier in section 3.2.1, the segmentation model we are using operates on two levels. The sentence embedding level is a two-layer bidirectional LSTM with an input size of 300 and a hidden size of 256, generating sentence representations through max-pooling over the outputs. The classifier level is similar to the first, but with double the input size (512), classifying segment boundaries based on labeled training data.
The model was trained using stochastic gradient descent (SGD) with a batch size of 32 over 20 epochs, using 100,000 documents from the Wiki727k dataset [14]. We optimized for cross-entropy loss, classifying whether a sentence marks the end of a segment. Early stopping was applied after 14 epochs, based on the validation loss plateauing.
In evaluating the text segmentation model, we used the 
p
k
 metric as defined by Beeferman et al. [3], which measures the probability of error when predicting segment boundaries at random points in the text. Simply put, a lower 
p
k
 score means the model is better at accurately identifying where segments end. The model achieved a 
p
k
 score of 35 on the WIKI-50 test set [14], compared to the original paper’s score of 20. However, it’s important to note that our focus here is not primarily on optimizing text segmentation but on enhancing retrieval through segmentation-chunking integration. As a result, we used a smaller dataset and fewer training epochs, while the original paper used much larger data and more intensive training. This trade-off allowed us to focus on the RAG framework while maintaining reasonable segmentation accuracy.
Clustering We set the 
k
-values to control the number of clusters, as outlined in Section 3.2.2, ensuring alignment with the average chunk size. For an average of 512 tokens, we used 
k
=
1.2
, for 1024 tokens, 
k
=
0.7
, and for 2048 tokens, 
k
=
0.4
. Lower 
k
-values directly reduce the number of clusters, resulting in larger token sizes per cluster.
4.4Retrieval Results
As shown in Table 2 and Table 3, our segmentation-clustering method performs better than the other chunking strategies across all three datasets. For NarrativeQA, the average 1024-token segment-cluster method achieves the highest ROUGE-L score of 26.54, outperforming both the base and semantic methods. Additionally, it shows an improvement in METEOR score, reaching 30.26. In the QASPER dataset, the 1024-token segment-cluster method again yields the best results, with an F1 score of 24.67. Similarly, in the QuALITY dataset, the segment-cluster method with an average of 512 tokens attains the highest accuracy of 63.77, outperforming the base 512-token method’s accuracy of 60.23.
While larger chunk sizes, such as the 2048-token configuration, might intuitively seem to provide more context, the results show diminishing returns in performance. This drop in scores is likely due to the increased size of each chunk, which makes the chunks more difficult to process and dilutes their coherence. As chunks grow larger, they capture more information but can become too broad, causing the reader model to lose focus on query-relevant details.
Table 2:Performance on QASPER and QuALITY: Evaluation of various chunking strategies on the QASPER and QuALITY datasets. The segmentation-clustering approach yields the highest F1 score on QASPER with 1024-token average segmentation and clustering, and the highest accuracy on QuALITY with 512-token average segmentation and clustering.
Chunk size
Top-k Chunks
Methods
F1 (QASPER)
Accuracy (QuALITY)
256
20
Base
19.28
58.16
Semantic
18.07
57.23
512
8
Base
20.33
60.23
Cluster Only
21.64
62.36
Segment + Cluster
21.95
63.77
1024
4
Base
22.07
58.23
Cluster Only
23.31
58.84
Segment + Cluster
24.67
59.08
2048
2
Base
22.05
57.54
Cluster Only
22.76
57.71
Segment + Cluster
23.89
58.85
Table 3:Performance on NarrativeQA: Comparison of various chunking strategies on the NarrativeQA dataset. The 1024-token average segmentation and clustering outperforms all other chunking strategies across all metrics.
Chunk size
Top-k Chunks
Methods
ROUGE-L
BLEU-1
BLEU-4
METEOR
256
20
Base
22.21
16.99
5.06
27.11
Semantic
22.5
16.55
5.51
26.56
512
8
Base
23.16
17.17
5.77
27.13
Cluster Only
24.12
17.91
6.55
27.56
Segment + Cluster
24.67
18.97
6.83
28.64
1024
4
Base
23.86
18.05
6.59
27.12
Cluster Only
25.15
19.28
6.97
29.05
Segment + Cluster
26.54
20.03
7.58
30.26
2048
2
Base
23.53
17.65
6.29
27.02
Cluster Only
25.67
19.13
6.80
29.64
Segment + Cluster
26.39
19.62
7.38
30.07
The Olympic Gene Pool

Question: The author believes that athletic ability changes over time mainly due to?

1. Top athletes having fewer children
2. Innate factors
3. Environment
4. Natural selection and genetics

512 Segment + Cluster: It is scarcely surprising that Ethiopian or Kenyan distance runners do better than everyone else …. Environmental differences between the two groups could account for differing levels of athletic success … Better health care and practicing condition affects athletic ability directly
Answer: 3


Base 512: We know that the inheritance of extra fingers or toes is determined genetically … Perhaps way, way back in human history, when our forebears were still fleeing saber-toothed tigers, natural selection for athletic prowess came into play… Indeed, the laws of natural selection probably work against athletes these days.
Answer: 2

Figure 4:Retrieved chunks based on multiple chunking strategy for the question about the story The Olympic Gene Pool.
We believed that using cohesive segments significantly improves the retrieval process by ensuring that meaningful units of text are retrieved. Traditional chunking often retrieves fragmented ideas due to arbitrary chunking, resulting in disjointed answers. Our Segment-Cluster method addresses this issue by grouping related segments, even if they are not adjacent. Fig. 4 shown that this approach captures broader themes, such as training and healthcare factors. While these factors may not seem relevant to the query at first glance, they together provide better context. As a result, our method retrieves more coherent and contextually relevant information, leading to improved accuracy and overall answer quality.
5Conclusion
This paper introduces a framework that integrates hierarchical text segmentation with retrieval-augmented generation (RAG) to improve the coherence and relevance of retrieved information. By combining segmentation and clustering in chunking, our method ensures that each chunk is semantically and contextually cohesive, addressing the limitations of traditional, fixed-length chunking.
Experiments showed our approach enhances retrieval accuracy and answering performance compared to traditional chunking. While a top-down segmentation approach could be ideal, current model limitations favor a bottom-up combination of supervised and unsupervised techniques. Future work may explore multi-level segmentation to streamline hierarchical representation in RAG or use enriched segments in knowledge graph construction to improve entity relationships and clustering accuracy.

