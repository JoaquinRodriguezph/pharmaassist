# Framework choice for answering (Step 5a): LangChain, LlamaIndex, or neither

Step 5 adds a function that takes a question, retrieves k = 5 chunks, asks the LLM to answer only from them with `[file.md#Section]` citations, and says so when the sources do not answer. Before writing it I checked whether LangChain or LlamaIndex should do that job. Labels: **Docs** = stated on the page named in "Sources" (read on 20 October 2026). **Measured** = printed by a command run on this machine. **Untested** = my expectation, not checked. **Not read** = I did not find or read a page on it, so I make no claim.

**Versions.** The pages I read show no current release numbers for either framework. The only versions they mention: `langchain-chroma>=0.1.2` in the install line, `langchain-google-genai` 4.0.0 and later using the `google-genai` SDK, and `llama-index-llms-gemini` discontinued as of 0.6.2. Both projects' docs change quickly, so this comparison is only as current as the read date.

I have installed neither framework. Nothing about how they behave with my code is Measured.

## 1. What each offers for this pipeline

| Piece | LangChain | LlamaIndex |
|---|---|---|
| Loading my four Markdown files | Not read | Not read |
| Splitting by heading | **Docs:** `MarkdownHeaderTextSplitter(headers_to_split_on)` splits on the headings you list and puts each heading's text in metadata under the name you give it. It strips the heading from the chunk text by default. It sets no size limit; for that, the page applies a second splitter (`RecursiveCharacterTextSplitter`) to each section, with overlap kept inside a section. The page says nothing about ids | **Docs:** `MarkdownNodeParser()` "parses raw markdown text"; the page does not say what metadata it adds or whether it limits size. `SentenceSplitter(chunk_size=..., chunk_overlap=...)` splits at sentence boundaries. Node ids are "automatically generated if not manually specified", and `TextNode(text=..., id_=...)` sets one by hand |
| Chroma wrapper | **Docs:** `langchain-chroma`, `Chroma(client=..., collection_name=..., embedding_function=...)`, `add_documents(documents, ids=...)`, `update_documents(ids=..., documents=...)`, `delete(ids=...)`, `similarity_search_by_vector(...)`, and a `filter=` dict. The reference page lists `embedding_function=None` and `collection_configuration=None` as defaults, but **does not say what happens when no embedding function is given**, and does not say whether the score from `similarity_search_with_score` is a distance or a similarity | **Docs:** `ChromaVectorStore(chroma_collection=...)` wraps a collection I create myself (including from `PersistentClient`), then `VectorStoreIndex.from_vector_store(vector_store, embed_model=...)`. Chroma is listed among the stores that keep "the entire index (vectors + text)" in the vector store. The page sets `embed_model` explicitly and does not name a default |
| My own embedder | **Docs:** the `Embeddings` interface is two methods, `embed_documents(texts)` and `embed_query(text)`. An adapter around my `Embedder` would be a few lines. **Untested** | **Docs:** subclass `BaseEmbedding` and implement `_get_query_embedding`, `_get_text_embedding`, `_get_text_embeddings` and the async versions. **Untested** |
| Prompt templates | Not read | Not read |
| Citations | Not read | **Docs:** `CitationQueryEngine.from_args(index, similarity_top_k=..., citation_chunk_size=512)` labels sources "Source 1", "Source 2", ... and the answer cites them as `[1]`. It **re-splits** retrieved nodes into citation chunks of `citation_chunk_size` (default 512) when that is smaller than the nodes. The page does not show the prompt it uses |
| Ollama | **Docs:** `langchain-ollama`, `ChatOllama(model=..., temperature=...)`, called with `invoke([("system", ...), ("human", ...)])`. Its feature table marks **token usage as not supported** | **Docs:** `llama-index-llms-ollama`, `Ollama(model=..., base_url="http://localhost:11434", temperature=None, context_window=-1, request_timeout=...)`. `context_window=-1` means "looked up from the model info". `complete` is a wrapper around `chat` |
| Gemini | **Docs:** `langchain-google-genai` (4.0.0 and later use the `google-genai` SDK), `ChatGoogleGenerativeAI(model=..., temperature=..., max_tokens=..., max_retries=...)`. It reads `GOOGLE_API_KEY`, then `GEMINI_API_KEY`. Token counts are in `usage_metadata`, with reasoning tokens under `output_token_details`. If temperature is unset on Gemini 3.0+, it becomes 1.0. LangChain's general model page says `max_retries` **defaults to 6** and retries 429 rate limits with exponential backoff. The Gemini page's examples set `max_retries=2` but do not state that integration's default | **Docs:** `llama-index-llms-gemini` is discontinued; use `llama-index-llms-google-genai`, `GoogleGenAI(model=...)`, which reads `GOOGLE_API_KEY` by default. Temperature is shown only inside a `generation_config`. The page does not mention retries or per-request token counts |

## 2. Replace or wrap my pieces?

What I must not lose:

- **Chunk ids:** `file.md#Fragment::n`, built with `slugify_heading`, and `section_id` = `file.md#Fragment`, which is the citation format the test set checks.
- **The fingerprints:** they make `ingest_kb` re-embed only what changed.
- **The embedder guard:** a store refuses an embedder whose name or dimension differs from the one it was built with.
- **The silent-default trap:** every Chroma write and read passes vectors explicitly, so Chroma's own default embedding function never runs.
- **`LLMClient`:** token counts, including Gemini's reasoning tokens as a separate field.
- **Retry policy:** it tells a per-minute 429 (wait and retry) from a daily-quota 429 (stop), because on the free tier of 20 requests per day, failed calls count.

| My piece | LangChain | LlamaIndex |
|---|---|---|
| `chunk_kb` | **Would wrap at best.** The header splitter does not produce my ids, my section ids, or my rule that the preamble and README are never indexed; I would compute all of that myself and copy it into `Document` metadata. Its size rule (a second splitter) is not my rule (paragraphs, then sentences, max 800 characters). **Untested** that the two give the same chunks; I expect they would not | **Would wrap at best.** I can set `id_` on each `TextNode` myself (Docs), so my ids could survive, but the parser would not create them. `MarkdownNodeParser`'s metadata and size behavior is not on the page I read |
| `ingest_kb` (fingerprints, upsert, delete) | **Would keep mine.** The wrapper has update and delete by id (Docs), but nothing I read compares fingerprints, so the "re-embed only what changed" logic stays mine | **Would keep mine.** The Chroma page updates and deletes through the raw `chroma_collection` (Docs), not through the index. How `ChromaVectorStore` lays out text and metadata in the collection is not on the page, so **reading my existing collection through it is Untested** and may need a re-ingest in its own format, which is where my fingerprints live |
| `retrieve` and the embedder guard | **Would keep mine.** Nothing I read checks the collection's recorded embedder against the one in use. The wrapper's behavior with `embedding_function=None` is undocumented, and that is the exact trap my store avoids. Its score is not documented as distance or similarity, while mine is `1 - distance`, Measured against my own cosine | **Would keep mine.** `embed_model` falls back to a global `Settings` default when not given (Docs); the page does not say what that default is, so forgetting it in one place is a risk I would have to guard against myself. **Untested** |
| `LLMClient` and retry | **Would wrap or lose information.** `ChatOllama` reports no token usage (Docs, feature table), and my Ollama client does. A generic default of 6 retries on 429 (Docs, models page), together with my earlier finding that failed calls count against the 20-a-day free tier, means I expect it to spend the Gemini quota on retries (Untested) unless I set `max_retries=0` and keep my own retry around it | **Would wrap or lose information.** No retry or token-usage behavior for `GoogleGenAI` on the page I read. It reads `GOOGLE_API_KEY` by default, and my Gemini client reads `GEMINI_API_KEY`, so it needs an explicit `api_key` |

**Summary of part 2:** in both frameworks, every piece I have to protect would stay my own code, with an adapter around it. The framework would add the call to the LLM and possibly a prompt template, which I expect to be a short function in my own code (Untested until I write it in 5b).

## 3. Choice: neither, for Step 5

**I keep my own code and add `app/rag/answer.py` on top of `retrieve` and `LLMClient`.** The reasons, from parts 1 and 2:

1. **The pieces that matter are mine either way.** Chunk ids, fingerprints, the embedder guard, the explicit-vectors rule and the quota-aware retry are not provided by either framework in what I read. Adopting one would mean writing adapters to keep them, plus a dependency that can change under me.
2. **The citation format is mine.** The test set checks `[file.md#Section]`. LlamaIndex's citation engine numbers sources and can re-split chunks into 512-token pieces (Docs). So I expect a `[1]` to point to a piece of a chunk that I would have to map back to its section id (Untested: an inference from the re-split, not something I ran). I need the model to cite section ids it was shown, and a check that every cited id was retrieved. That is a prompt builder and a parser: small, testable code.
3. **Two documented behaviors work against me.** `ChatOllama` reports no token usage (Docs). LangChain's general model page says `max_retries` defaults to 6 and that 429 responses are retried (Docs). My own earlier finding is that failed calls count against the 20-requests-a-day free tier. Together, I expect that default would spend the Gemini quota on retries (Untested). Both can be worked around, but only by switching the framework's features off.
4. **Step 5 needs a prompt builder and a citation check,** which I expect to be short in my own code (Untested until 5b).

**What would change my mind:**

- needing loaders for formats I do not handle (PDF, web pages);
- tool routing in a later step, where a framework's tool-calling support could save real work;
- a second vector store in production.

I would re-check at that point, with an install and a measured comparison, not from docs alone.

**What I can claim:** I read the current docs of both on the pieces above. I did not install or run either, so every statement about how they would behave with my store is Untested.

**What this doesn't cover.** I will not have hands-on LangChain or LlamaIndex experience from this project, and many AI Engineer job postings list one of them. A cheap way to close part of that gap later, not now: spend about 30 minutes wrapping my `retrieve` in a LangChain retriever as a measured experiment, and record what it took and what it changed.

## Sources

- [LangChain: Chroma vector store](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [LangChain reference: Chroma class](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma)
- [LangChain: Markdown header splitter](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)
- [LangChain: Embedding models](https://docs.langchain.com/oss/python/integrations/text_embedding)
- [LangChain: ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [LangChain: ChatGoogleGenerativeAI](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
- [LangChain: Models (retries, usage metadata)](https://docs.langchain.com/oss/python/langchain/models)
- [LlamaIndex: Chroma vector store](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/chromaindexdemo/)
- [LlamaIndex: Customizing storage](https://developers.llamaindex.ai/python/framework/module_guides/storing/customization/)
- [LlamaIndex: Node parser modules](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/)
- [LlamaIndex: Using nodes](https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_nodes/index.md)
- [LlamaIndex: Settings](https://developers.llamaindex.ai/python/framework/module_guides/supporting_modules/settings/index.md)
- [LlamaIndex: Custom embeddings](https://developers.llamaindex.ai/python/framework/integrations/embeddings/custom_embeddings/)
- [LlamaIndex: Ollama LLM reference](https://developers.llamaindex.ai/python/framework-api-reference/llms/ollama/)
- [LlamaIndex: Gemini (discontinued) and Google GenAI](https://developers.llamaindex.ai/python/framework/integrations/llm/google_genai/)
- [LlamaIndex: Using LLMs](https://developers.llamaindex.ai/python/framework/understanding/using_llms/)
- [LlamaIndex: Citation query engine](https://developers.llamaindex.ai/python/examples/query_engine/citation_query_engine/)