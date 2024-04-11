print("hello world")

"""
# Multi-Document Agents (V1)

In this guide, you learn towards setting up a multi-document agent over the LlamaIndex documentation.

This is an extension of V0 multi-document agents with the additional features:
- Reranking during document (tool) retrieval
- Query planning tool that the agent can use to plan


We do this with the following architecture:

- setup a "document agent" over each Document: each doc agent can do QA/summarization within its doc
- setup a top-level agent over this set of document agents. Do tool retrieval and then do CoT over the set of tools to answer a question.
"""

"""
%pip install llama-index-core
%pip install llama-index-agent-openai
%pip install llama-index-readers-file
%pip install llama-index-postprocessor-cohere-rerank
%pip install llama-index-llms-openai
%pip install llama-index-embeddings-openai
%pip install unstructured[html]
!pip install nltk
!pip install llama_index.readers.file
!pip install llama-index-question-gen-openai
!pip install PyMuPDF

%load_ext autoreload
%autoreload 2

"""

#In this section, we'll load in the LlamaIndex documentation.




from google.colab import drive
drive.mount('/content/drive')

import nltk
nltk.download('averaged_perceptron_tagger')

from llama_index.readers.file import UnstructuredReader

reader = UnstructuredReader()

from pathlib import Path

# Path to your PDF files
pdf_files_path = Path("/content/drive/MyDrive/MMMCZCS")

# Gathering all PDF files
all_pdf_files = list(pdf_files_path.rglob("*.pdf"))

print(f"Total PDF files found: {len(all_pdf_files)}")

from llama_index.core import Document
import fitz  # PyMuPDF



# TODO: set to higher value if you want more docs
doc_limit = 30



docs = []
for idx, pdf_path in enumerate(all_pdf_files):
    if idx >= doc_limit:  # Adjusted to >= to process exactly doc_limit files
        break
    print(f"Processing file {idx+1}/{min(len(all_pdf_files), doc_limit)}: {pdf_path}")

    # Opening the PDF file
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"  # Extracting text from each page

    # Creating Document
    loaded_doc = Document(
        text=text,
        metadata={"path": str(pdf_path)},
    )
    docs.append(loaded_doc)

print(f"Processed {len(docs)} documents.")

"""Define Global LLM + Embeddings"""

import os

os.environ["OPENAI_API_KEY"] = "key"

import nest_asyncio

nest_asyncio.apply()

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

llm = OpenAI(model="gpt-3.5-turbo")
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", embed_batch_size=256
)

"""## Building Multi-Document Agents

In this section we show you how to construct the multi-document agent. We first build a document agent for each document, and then define the top-level parent agent with an object index.

### Build Document Agent for each Document

In this section we define "document agents" for each document.

We define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an OpenAI function calling agent.

This document agent can dynamically choose to perform semantic search or summarization within a given document.

We create a separate document agent for each city.
"""

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
import os
from tqdm.notebook import tqdm
import pickle


async def build_agent_per_doc(nodes, file_base):
    print(file_base)

    vi_out_path = f"./data/llamaindex_docs/{file_base}"
    summary_out_path = f"./data/llamaindex_docs/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data/llamaindex_docs/").mkdir(parents=True, exist_ok=True)
        # build vector index
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=vi_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
        )

    # build summary index
    summary_index = SummaryIndex(nodes)

    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", llm=llm
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            await summary_query_engine.aquery(
                "Extract a concise 3-4 line summary of this document"
            )
        )
        pickle.dump(summary, open(summary_out_path, "wb"))
    else:
        summary = pickle.load(open(summary_out_path, "rb"))

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-4-0125-preview	")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the `{file_base}.pdf` part of the Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent, summary


async def build_agents(docs):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    # # this is for the baseline
    # all_nodes = []

    for idx, doc in enumerate(tqdm(docs)):
        nodes = node_parser.get_nodes_from_documents([doc])
        # all_nodes.extend(nodes)

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = await build_agent_per_doc(nodes, file_base)

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict

agents_dict, extra_info_dict = await build_agents(docs)

"""### Build Retriever-Enabled OpenAI Agent

We build a top-level agent that can orchestrate across the different document agents to answer any user query.

This `RetrieverOpenAIAgent` performs tool retrieval before tool use (unlike a default agent that tries to put all tools in the prompt).

**Improvements from V0**: We make the following improvements compared to the "base" version in V0.

- Adding in reranking: we use Cohere reranker to better filter the candidate set of documents.
- Adding in a query planning tool: we add an explicit query planning tool that's dynamically created based on the set of retrieved tools.

"""

# define tool for each document agent
all_tools = []
for file_base, agent in agents_dict.items():
    summary = extra_info_dict[file_base]["summary"]
    doc_tool = QueryEngineTool(
        query_engine=agent,
        metadata=ToolMetadata(
            name=f"tool_{file_base}",
            description=summary,
        ),
    )
    all_tools.append(doc_tool)

print(all_tools[0].metadata)

# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import (
    ObjectIndex,
    ObjectRetriever,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.llms.openai import OpenAI


llm = OpenAI(model_name="gpt-4-0613")

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
vector_node_retriever = obj_index.as_node_retriever(
    similarity_top_k=10,
)


# define a custom object retriever that adds in a query planning tool
class CustomObjectRetriever(ObjectRetriever):
    def __init__(
        self,
        retriever,
        object_node_mapping,
        node_postprocessors=None,
        llm=None,
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OpenAI("gpt-4-0613")
        self._node_postprocessors = node_postprocessors or []

    def retrieve(self, query_bundle):
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_str=query_bundle)

        nodes = self._retriever.retrieve(query_bundle)
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, llm=self._llm
        )
        sub_question_description = f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents. Give an answer that is 8-9 lines.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]

os.environ["COHERE_API_KEY"] = "key"

# wrap it with ObjectRetriever to return objects
custom_obj_retriever = CustomObjectRetriever(
    vector_node_retriever,
    obj_index.object_node_mapping,
    node_postprocessors=[CohereRerank(top_n=5)],
    llm=llm,
)

tmps = custom_obj_retriever.retrieve("hello")

# should be 5 + 1 -- 5 from reranker, 1 from subquestion
print(len(tmps))

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent

top_agent = OpenAIAgent.from_tools(
    tool_retriever=custom_obj_retriever,
    system_prompt=""" \
You are an agent designed to answer queries about the zero carbon shipping and Mærsk McKinney Møller Center for Zero Carbon Shipping.
Please always use the tools provided to answer a question. Answer in 8-9 lines. Do not rely on prior knowledge.\

""",
    llm=llm,
    verbose=True,
)


"""### Define Baseline Vector Store Index

As a point of comparison, we define a "naive" RAG pipeline which dumps all docs into a single vector index collection.

We set the top_k = 4
"""

all_nodes = [
    n for extra_info in extra_info_dict.values() for n in extra_info["nodes"]
]

base_index = VectorStoreIndex(all_nodes)
base_query_engine = base_index.as_query_engine(similarity_top_k=4)

"""## Running Example Queries

Let's run some example queries, ranging from QA / summaries over a single document to QA / summarization over multiple documents.
"""

response = top_agent.query(
    "What are LCA calculations?",
)

print(response)

# baseline
response = base_query_engine.query(
    "What is zero carbon shipping?",
)
print(str(response))

