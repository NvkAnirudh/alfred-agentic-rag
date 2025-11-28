# Learnings: Building an Agentic RAG System with Smolagents

## Journey Overview
This document captures key learnings from building an agentic RAG system using Smolagents, from initial implementation through optimization attempts.

---

## 1. Why Convert HuggingFace Datasets to Document Objects?

### Question
Why do we need to convert HuggingFace datasets into `Document` objects?

### Answer
RAG frameworks (LangChain, LlamaIndex) expect a standardized format:
- **`page_content`**: The searchable text content
- **`metadata`**: Additional information (source, ID, etc.)

**Benefits:**
- ✅ Works seamlessly with framework vector stores and retrievers
- ✅ Preserves metadata for filtering and source attribution
- ✅ Enables use of built-in embedding and retrieval features
- ✅ Avoids writing custom retrieval logic for each framework

**Example:**
```python
# HuggingFace dataset entry (raw dict)
{'name': 'Ada Lovelace', 'bio': 'Mathematician', 'email': 'ada@example.com'}

# Converted to Document
Document(
    page_content='Ada Lovelace: Mathematician',
    metadata={'name': 'Ada Lovelace', 'email': 'ada@example.com'}
)
```

---

## 2. List vs Tuple in `str.join()`

### Question
Why use `[]` (list) instead of `()` (tuple) for `str.join()`?

### Answer
**Both work**, but lists are preferred:

```python
# Both valid
page_content = "\n".join([...])  # ✅ Convention
page_content = "\n".join((...))  # ✅ Also works
```

**Why lists are better:**
- ✅ Standard convention for collections
- ✅ More readable (clear "list of items")
- ✅ Mutable - can modify before joining if needed
- ✅ No performance difference for small collections

---

## 3. Tool Initialization: `super().__init__()` and `is_initialized`

### Question
Why call `super().__init__()` and set `is_initialized = True`?

### Initial Understanding (Incorrect)
We thought these were **required** for Smolagents tools to work.

### Actual Reality
**Tools work without them!** We tested and confirmed:
- ✅ Tool instantiation works
- ✅ Direct `forward()` calls work
- ✅ Agent integration works
- ✅ No errors in current Smolagents version

### Why They're Still Recommended (Best Practices)
1. **Future compatibility** - Smolagents may add stricter checks later
2. **Hub sharing** - Required for publishing to HuggingFace Hub
3. **Code clarity** - Signals initialization is complete
4. **Python conventions** - Proper inheritance pattern

**Conclusion:** Not strictly required, but good practice.

---

## 4. LangChain Retriever API: `invoke()` vs `get_relevant_documents()`

### The Error
```python
AttributeError: 'BM25Retriever' object has no attribute 'get_relevant_documents'
```

### The Fix
```python
# Old API (deprecated)
results = self.retriever.get_relevant_documents(query)  # ❌

# New API (LCEL)
results = self.retriever.invoke(query)  # ✅
```

### Why This Matters
LangChain migrated to **LangChain Expression Language (LCEL)**:
- ✅ Consistent interface across all components
- ✅ Works with chains and pipelines
- ✅ Future-proof
- ✅ Better type safety

---

## 5. Agent Error Recovery and Fallback Behavior

### Observation
When the retriever tool failed, the agent still answered the question using its own knowledge.

### What Happened
```
Step 1: Tool call failed (AttributeError)
Step 2: Agent used built-in knowledge to answer
Result: Generic answer about Ada Lovelace (not from dataset)
```

### Key Learning
**Agentic systems are resilient:**
- ✅ Don't crash on tool failures
- ✅ Adapt and find alternative approaches
- ✅ Multi-step planning and recovery

### The Problem for RAG
This is actually **bad for RAG systems**:
- ❌ Want specific dataset information, not generic knowledge
- ❌ Should report "I don't have that information" if tool fails
- ❌ Defeats the purpose of retrieval-augmented generation

---

## 6. Model Selection: Speed vs Quality Trade-offs

### Default Model Performance
**Model:** `Qwen/Qwen3-Next-80B-A3B-Thinking` (InferenceClientModel default)

**Results:**
- ✅ Perfect quality - natural responses
- ✅ No tool hallucination
- ✅ Only 2 steps
- ❌ Slow: 28 seconds total (6.5s + 21.2s)

### Optimization Attempt 1: Qwen/Qwen2.5-72B-Instruct

**Goal:** Reduce latency

**Results:**
- ✅ Faster: 12 seconds (vs 28 seconds)
- ❌ Tool hallucination (tried `wikipedia_search`, `web_search`)
- ❌ Raw output (dumped all 3 retrieved docs)
- ❌ 4 steps instead of 2

**Why hallucination happened:**
- Model trained on common tool patterns
- "Knows" Wikipedia/web search are common for finding people
- Assumed these tools existed even though not provided

### Optimization Attempt 2: ToolCallingAgent

**Goal:** Prevent tool hallucination

**Changes:**
```python
# Before
alfred = CodeAgent(tools=[guest_info_tool], model=model)

# After
alfred = ToolCallingAgent(
    tools=[guest_info_tool], 
    model=model,
    instructions="Provide natural, conversational responses..."
)
```

**Results:**
- ✅ No tool hallucination
- ✅ Natural language response (thanks to instructions)
- ⚠️ JSON parsing errors (model not optimized for tool calling)
- ⏱️ 14 seconds (faster than default, but retries due to errors)

**Why parsing errors:**
- `ToolCallingAgent` expects structured JSON output
- `Qwen/Qwen2.5-72B-Instruct` not trained for tool calling format
- Model sometimes outputs JSON correctly, sometimes doesn't
- Agent retries until valid JSON received

---

## 7. CodeAgent vs ToolCallingAgent

### CodeAgent
**How it works:** Generates Python code to call tools

**Pros:**
- ✅ Flexible - can use any Python function
- ✅ No JSON parsing requirements
- ✅ Works with any model

**Cons:**
- ❌ Can hallucinate tools (generates code for non-existent functions)
- ❌ Security concerns (executes arbitrary code)

### ToolCallingAgent
**How it works:** Uses structured JSON for tool calls

**Pros:**
- ✅ Prevents hallucination (must choose from available tools)
- ✅ More secure (no arbitrary code execution)
- ✅ Better for production

**Cons:**
- ❌ Requires models trained for tool calling (GPT-4, Claude, Gemini)
- ❌ JSON parsing errors with non-optimized models
- ❌ Less flexible

---

## 8. Model Compatibility with Tool Calling

### Models That Work Well with ToolCallingAgent
- ✅ **OpenAI:** GPT-4, GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- ✅ **Anthropic:** Claude models
- ✅ **Google:** Gemini models
- ✅ **Qwen:** Qwen3-Next-80B-A3B-Thinking (default)

### Models That Have Issues
- ⚠️ **Qwen/Qwen2.5-72B-Instruct** - Not optimized for structured tool calling
- ⚠️ Most general instruction-following models

### Non-existent Models (Mistakes Made)
- ❌ `Qwen/Qwen2.5-72B-Instruct-Turbo` - Doesn't exist!

---

## 9. Latency Optimization Strategies

### Strategy 1: Use Faster Models
```python
# Slow but high quality (28s)
model = InferenceClientModel()  # Default: 80B-Thinking

# Faster alternatives
model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")  # 12s
model = InferenceClientModel(model_id="Qwen/Qwen2.5-32B-Instruct")  # 8s
model = InferenceClientModel(model_id="Qwen/Qwen2.5-7B-Instruct")   # 5s
```

**Trade-off:** Speed vs quality/reliability

### Strategy 2: Use OpenAI Models (Best)
```python
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

**Results:**
- ✅ Fastest: 3-5 seconds
- ✅ Best quality
- ✅ No parsing errors
- 💰 Costs ~$0.0001 per query

### Strategy 3: Optimize Tool Output
Return only the most relevant result instead of top 3:
```python
def forward(self, query: str):
    results = self.retriever.invoke(query)
    if results:
        return results[0].page_content  # Only first result
    return "No information found"
```

### Strategy 4: Add Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def forward(self, query: str):
    # Cache repeated queries
    ...
```

### Strategy 5: Use Streaming
```python
ToolCallingAgent(tools=[guest_info_tool], model=model, instructions="""....""", stream_outputs=True)
# User sees results immediately
for chunk in response:
    print(chunk, end="", flush=True)
```

---

## 10. System Prompts and Instructions

### Incorrect Parameter Name
```python
# ❌ Error: unexpected keyword argument 'system_prompt'
alfred = ToolCallingAgent(
    tools=[guest_info_tool],
    model=model,
    system_prompt="..."  # Wrong parameter name!
)
```

### Correct Parameter Name
```python
# ✅ Correct: use 'instructions'
alfred = ToolCallingAgent(
    tools=[guest_info_tool],
    model=model,
    instructions="""You are Alfred, a helpful assistant.
    Provide natural, conversational responses based on retrieved information."""
)
```

### Impact of Instructions
Without instructions:
- ❌ Returns raw retrieved documents

With instructions:
- ✅ Synthesizes natural language responses
- ✅ More conversational and helpful

---

## 11. Final Recommendations

### For Production RAG Systems

1. **Use OpenAI Models** (gpt-4o-mini)
   - Best balance of speed, quality, and reliability
   - ~$0.0001 per query (very affordable)
   - 3-5 second latency

2. **Use ToolCallingAgent**
   - Prevents tool hallucination
   - More secure than CodeAgent
   - Better for production

3. **Add Clear Instructions**
   - Guide the agent's behavior
   - Ensure natural language responses
   - Specify desired output format

4. **Optimize Tool Output**
   - Return only most relevant results
   - Format responses naturally
   - Include metadata when useful

5. **Monitor and Log**
   - Track latency per step
   - Monitor token usage
   - Log tool calls and errors

### For Free/Open-Source Models

1. **Use Default Model** for best quality
   - `InferenceClientModel()` (Qwen3-Next-80B-A3B-Thinking)
   - Accept 28-second latency for high quality

2. **Or Accept Trade-offs** with faster models
   - Potential hallucination
   - Parsing errors
   - Lower quality responses

---

## 12. Key Takeaways

1. **Agentic systems are resilient but unpredictable**
   - They adapt when tools fail
   - May use fallback knowledge instead of admitting ignorance
   - Need careful prompt engineering

2. **Model selection is critical**
   - Different models have different strengths
   - Tool calling requires specific training
   - Speed vs quality is a real trade-off

3. **Framework APIs evolve**
   - LangChain moved from `get_relevant_documents()` to `invoke()`
   - Always check current documentation
   - Use LCEL for future compatibility

4. **Best practices matter for production**
   - Even if `super().__init__()` isn't required now
   - Even if `is_initialized` isn't checked now
   - Future versions may enforce these

5. **Free models have limitations**
   - Slower than commercial APIs
   - Less reliable tool calling
   - May hallucinate tools
   - Worth it for learning, but consider paid options for production

---

## Final Implementation Choice

**Going with:** `gpt-4o-mini` via OpenAI

**Rationale:**
- ✅ Best latency (3-5 seconds)
- ✅ Excellent quality
- ✅ Reliable tool calling
- ✅ No parsing errors
- ✅ No hallucination
- 💰 Extremely affordable (~$0.0001 per query)

**Trade-off accepted:** Small cost for significantly better user experience
