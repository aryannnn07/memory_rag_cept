# Generative AI Course Notes

## Topic 1: Transformers
The Transformer architecture was introduced in the 2017 paper "Attention Is All You Need". It replaced recurrence (RNNs, LSTMs, GRUs) with self-attention, which lets every word look at every other word in a sentence at the same time. This parallel processing was the biggest change: training became much faster on GPUs, and long-range dependencies stopped fading the way they do in an RNN's hidden state.

Self-attention works with three vectors per token: Query, Key and Value. The query asks "what am I looking for?", the key says "what do I contain?", and the value carries the actual information. Attention is computed as softmax(Q K transpose divided by square root of d_k) multiplied by V. Dividing by the square root of d_k keeps the dot products small so softmax does not saturate and gradients stay useful.

Multi-head attention runs several attention heads in parallel, each with its own learned weight matrices, so different heads can capture different relationships such as syntax, coreference and position. The original model used 8 heads of 64 dimensions each. Because attention has no built-in sense of order, positional encodings built from sine and cosine functions are added to the token embeddings. Each block also uses residual connections plus layer normalisation ("Add & Norm") and a position-wise feed-forward network.

The encoder uses unmasked self-attention. The decoder uses masked self-attention so a position cannot see future tokens, plus cross-attention where queries come from the decoder and keys and values come from the encoder output.

Temperature is an inference-time setting that divides the logits before softmax. A low temperature such as 0.2 makes output focused and nearly deterministic. A temperature of 1.0 keeps the model's learned distribution. A high temperature such as 1.5 flattens the distribution and makes output more creative but less reliable.

## Topic 2: Types of Transformer Models
Encoder-only models such as BERT see the whole sentence in both directions. BERT is pre-trained with Masked Language Modelling, where about 15 percent of tokens are hidden and predicted from context, and Next Sentence Prediction. BERT is best for understanding tasks: classification, named entity recognition and extractive question answering. For classification a small head is placed on the [CLS] token output. Variants include RoBERTa, DistilBERT and ALBERT.

Decoder-only models such as GPT, LLaMA, Qwen and Claude generate text left to right using causal (masked) attention. They are pre-trained with next-token prediction. At large scale they show in-context learning: they can perform a new task from zero-shot instructions or a few examples in the prompt without fine-tuning. This flexibility is why modern LLMs are decoder-only.

Encoder-decoder models such as T5 and BART use the full architecture. T5 frames every task as text-to-text with a prefix like "summarize:" or "translate English to French:". BART is pre-trained as a denoising autoencoder and is strong at summarisation. These models suit translation and summarisation.

## Topic 3: Large Language Models
An LLM is a large decoder-only Transformer trained on trillions of tokens. Training happens in three stages. Stage 1, pre-training, produces a base model that completes text but does not reliably follow instructions. Stage 2, supervised fine-tuning or instruction tuning, trains on instruction and response pairs so the model follows requests. Stage 3, RLHF, collects human preference comparisons, trains a reward model on them, and optimises the LLM with PPO while a KL penalty keeps it close to the SFT model. DPO, Direct Preference Optimisation, is a simpler alternative that learns from preference pairs directly without a separate reward model.

On Hugging Face, base models have names like Llama-3.1-8B while chat-ready models carry an -Instruct or -Chat suffix. Instruct models expect a chat template with system, user and assistant roles.

## Topic 4: Hugging Face
Hugging Face hosts hundreds of thousands of models and datasets. The transformers library offers pipeline() for quick use, for example pipeline("sentiment-analysis"), and Auto classes such as AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM and AutoModelForSeq2SeqLM for more control. Tokenizers turn text into token IDs using subword methods: WordPiece for BERT, byte-level BPE for GPT-2 and LLaMA, and SentencePiece for T5. The datasets library loads data in one line, for example load_dataset("emotion"). Always use tokenizer.apply_chat_template for instruct models.

## Topic 5: Using LLMs through APIs
Most providers use the OpenAI chat completions format, so the same code works for OpenAI, Groq and OpenRouter by changing the base_url, API key and model name. Groq runs open models on fast custom hardware and has a free tier. OpenRouter gives one key for many providers. API keys belong in a .env file loaded with python-dotenv and listed in .gitignore. Streaming returns tokens as they are generated. FastAPI can wrap an LLM behind endpoints with Pydantic request and response models, and it generates interactive docs at /docs. Rate limits are handled with retries and exponential backoff.

## Topic 6: Running LLMs Locally with Ollama
Ollama runs models locally with a client-server design: a server on localhost port 11434 loads models, and the CLI, Python library and HTTP clients talk to it. Key commands are ollama list, ollama run, ollama pull, ollama ps and ollama stop. Ollama uses quantised models: a 3 billion parameter model needs about 12 GB in 32-bit floats but under 2 GB at 4-bit, which is why it fits on an 8 GB laptop. Ollama also exposes an OpenAI-compatible endpoint at http://localhost:11434/v1, and a Modelfile can set a system prompt and parameters.

## Topic 7: Inference Code Structure
Inference follows Instruction, Input, Output, which maps to the system, user and assistant roles. A strong instruction specifies role, task, output format, constraints and context. Few-shot prompting adds worked examples to the messages. Wrapping this pattern in a reusable class keeps the instruction fixed while inputs change, and batch inference can run many inputs concurrently while respecting rate limits.

## Topic 8: Fine-Tuning
Full fine-tuning updates every parameter and needs roughly 16 bytes per parameter for weights, gradients and Adam optimiser states, about 48 GB for a 3B model. LoRA, Low-Rank Adaptation, freezes the original weights and trains two small matrices A and B per targeted layer, so the update is W x plus B A x. The rank r controls how many parameters are trained, commonly 4 to 64, often under 1 percent of the model. QLoRA loads the frozen base model in 4-bit NF4 precision and trains LoRA adapters in higher precision, cutting memory further. The PEFT library implements LoRA and QLoRA, and only the small adapter needs to be saved.

## Topic 9: Retrieval-Augmented Generation
RAG retrieves relevant text at query time and adds it to the prompt, which addresses knowledge cutoffs, gives access to private documents and reduces hallucination without changing model weights. Indexing splits documents into overlapping chunks, embeds each chunk with an encoder model such as all-MiniLM-L6-v2, and stores the vectors in a vector database such as ChromaDB. At query time the question is embedded, the most similar chunks are found with cosine similarity, and the LLM answers using only that context. Typical settings are chunks of about 500 characters with some overlap and a top_k of 3 to 5. A RAG system is evaluated on retrieval quality, faithfulness to the context and answer relevance.

## Memory RAG
Memory RAG extends classic RAG with memory about the user and the conversation. Classic RAG is stateless: every question is answered as if from a stranger. Memory RAG keeps short-term memory, the last few messages of the current session, and long-term memory, durable facts about the user such as goals, preferences, deadlines and topics they struggle with, stored as vectors in their own ChromaDB collection.

On every turn Memory RAG runs a read path and a write path. The read path recalls memories by similarity and re-ranks them with a score that combines relevance, recency and importance. The recalled memories personalise the answer and also guide retrieval, by rewriting a vague question into more specific search clues. The write path asks the LLM to extract durable facts from the exchange and to choose an operation for each: ADD a new memory, UPDATE an existing one when facts change, or DELETE one when the user corrects it or asks to forget it. Near-duplicate memories are skipped.

Memories fade when they are not used: recency decays exponentially with the hours since a memory was last accessed, and low-importance memories that go unused can be pruned. Recalling a memory refreshes it. Users should be able to see and delete what is stored about them.
