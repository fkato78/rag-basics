# RAG BASICS

- More context is not always better.
- Better context is better.

- metadata: Use metadata to control retrieval, and use selected metadata to explain retrieval.
- Citations
  - Use metadata for filtering & ranking during retrieval, then inject a subset of that metadata (source, page, date) as citations in the final output.

- Reranking (natural next step after retrieval)
- Agent Loops (Tool Calling on steroids)
- Memory (short-term vs long-term memory)

```sh
RAG → retrieve knowledge
Memory → retrieve user context
Tools → interact with systems
Agent → orchestrate everything

```

- Evaluation (how to measure RAG quality systematically)

```sh
Evaluation is NOT: Did GPT answer correctly?
Evaluation starts with: Did retrieval retrieve the correct information?
```

- Hybrid Search (vector search + keyword/BM25)

```sh
Vector Search = Meaning
BM25 = Exact Terms
Hybrid Search = Meaning + Exact Terms

```

- Current RAG: NO Hybrid Search:

```sh
Question
↓
Embedding
↓
Vector Search
↓
Top K
↓
LLM
```

- RAG with Hybrid:

```sh
Question
↓
Vector Search
↓
Top K

+

Keyword Search
↓
Top K

↓

Merge Scores

↓

Best Results

↓

LLM
```

## AI Summary

```sh
LLM
│
├── Prompting
│
├── Structured Output
│
├── Tool Calling
│
├── RAG
│   ├── Embeddings
│   ├── Vector DB
│   ├── Chunking
│   ├── Metadata
│   ├── Evaluation
│   ├── Reranking (optional)
│   └── Hybrid Search (optional)
│
├── Memory (optional)
│
└── Agent Loops

```

```sh
Embeddings = Convert meaning into numbers

RAG = Search before answering

Memory = Remember user information

Tool Calling = Call a function

Agent = Tool calling in a loop

Hybrid Search = Vector search + keyword search

Reranking = Reorder retrieved results

```

```sh
                     User
                       |
                       v

                +-------------+
                |     LLM     |
                +-------------+
                  ^    ^    ^
                  |    |    |
                  |    |    |
             Memory   RAG  Tools
                  |    |    |
                  v    v    v

         User Data  Knowledge  Actions

                       |
                       v

                    Answer
```

## Important to know

```sh
Framework Feature      Your Mental Model
------------------------------------------------
Retriever          ->  RAG
Memory             ->  Context Retrieval
Tool               ->  Function Calling
Agent              ->  Tool Calling Loop
Chain              ->  Function Composition

```
