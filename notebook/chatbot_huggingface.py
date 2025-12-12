import google.generativeai as genai
import psycopg
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configuration de la connexion à la base de données
db_connection_str = "dbname=rag_chatbot user=postgres password=yesmine123 host=localhost port=5433"

# Clé API Gemini
GEMINI_API_KEY = "AIzaSyDfsyGoW_jjxj8-0RemaGAIoEIGkCl-rj0"

# Initialiser Gemini pour les embeddings uniquement
genai.configure(api_key=GEMINI_API_KEY)

# ==================== CHOIX DU MODÈLE ====================
# Options de modèles Hugging Face (du plus petit au plus grand):
# 1. "microsoft/DialoGPT-medium" - Petit, rapide, conversationnel (355M params)
# 2. "mistralai/Mistral-7B-Instruct-v0.2" - Performant (7B params) ⭐ RECOMMANDÉ
# 3. "meta-llama/Llama-2-7b-chat-hf" - Excellent (7B params, nécessite token HF)
# 4. "google/flan-t5-large" - Bon compromis (780M params)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Changez selon vos besoins

# ------------------
# Helpers: embeddings -> DB search -> prompt -> generate
# ------------------


def embed_text(text: str, model_name: str = "models/text-embedding-004") -> list[float]:
	"""Return embedding vector for a single text using Gemini.

	This expects the same embedding model used to populate your DB.
	"""
	if not text or not text.strip():
		raise ValueError("text is empty")

	resp = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
	# most genai responses use a dict with key 'embedding'
	if isinstance(resp, dict) and "embedding" in resp:
		return resp["embedding"]
	# fallback common structure
	try:
		return resp["data"][0]["embedding"]
	except Exception as exc:
		raise RuntimeError("Unable to extract embedding from response: %r" % (resp,)) from exc


def fetch_similar_from_db(query_embedding: list[float], top_k: int = 3, connection: str = db_connection_str):
	"""Fetch top-k similar documents from the embeddings table using pgvector syntax.

	Returns list of tuples (id, corpus, similarity).
	"""
	if not query_embedding:
		return []

	with psycopg.connect(connection) as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				SELECT id, corpus,
					   1 - (embedding <=> %s::vector) AS similarity
				FROM embeddings
				ORDER BY embedding <=> %s::vector
				LIMIT %s
				""",
				(query_embedding, query_embedding, top_k),
			)
			return cur.fetchall()


def build_prompt(question: str, docs: list[tuple], max_chars: int = 1500) -> str:
	"""Build a simple RAG-style prompt that provides context then asks the question.

	- docs is list of (id, corpus, similarity)
	- We truncate total context size to max_chars for safety.
	"""
	context_parts = []
	used = 0
	for _id, content, _sim in docs:
		if not content:
			continue
		part = content.strip()
		if used + len(part) > max_chars:
			part = part[: max(0, max_chars - used)]
		context_parts.append(part)
		used += len(part)
		if used >= max_chars:
			break

	context_text = "\n\n---\n\n".join(context_parts)

	prompt = (
		"You are a helpful assistant. Use the following retrieved documents to answer the question.\n\n"
		"If the answer is not contained in the documents, say 'I don't know'. Do not hallucinate.\n\n"
		"CONTEXT:\n" + context_text + "\n\nQUESTION: " + question + "\n\nAnswer:" 
	)
	return prompt


def load_qwen_model(model_name: str = MODEL_NAME, device: str | int | None = None):
	"""Load tokenizer and causal LM for Qwen (or other HF model). Returns (tokenizer, model, gen_fn).

	By default this uses device=0 if CUDA available, else CPU. Use pipeline() wrapper for generation.
	"""
	# Resolve device
	if device is None:
		device = 0 if torch.cuda.is_available() else "cpu"

	# Trust remote code for some models (Qwen/other custom Hub code)
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

	# Only use device_map / low_cpu_mem_usage when accelerate is available
	try:
		import accelerate  # type: ignore
		accelerate_available = True
	except Exception:
		accelerate_available = False

	use_cuda = torch.cuda.is_available()

	# If accelerate is available and CUDA present, allow device_map auto placement
	if accelerate_available and use_cuda:
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			trust_remote_code=True,
			device_map="auto",
			low_cpu_mem_usage=True,
		)
		pipeline_device = 0
	else:
		# Fallback: load model without device_map (CPU or GPU single-device)
		# This avoids the accelerate dependency on machines without it.
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			trust_remote_code=True,
		)
		pipeline_device = 0 if use_cuda else -1

	gen = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
		device=pipeline_device,
	)

	return tokenizer, model, gen


def generate_answer(question: str, top_k: int = 3, model_name: str = MODEL_NAME):
	"""High level: embed question, fetch contexts, build prompt and generate answer with Qwen.

	Returns dictionary with answer and used contexts for traceability.
	"""
	q_emb = embed_text(question)
	docs = fetch_similar_from_db(q_emb, top_k=top_k)
	prompt = build_prompt(question, docs)

	# Lazy-load model/tokenizer for simplicity
	tokenizer, model, gen = load_qwen_model(model_name)

	# generation parameters: keep small sensible defaults
	gen_kwargs = dict(max_length=512, temperature=0.7, top_p=0.9, do_sample=True, num_return_sequences=1)

	out = gen(prompt, **gen_kwargs)
	text = out[0]["generated_text"] if isinstance(out, list) and out else str(out)

	return {"answer": text, "contexts": docs, "prompt": prompt}


# ------------------
# CLI: run a single question interactively (or call generate_answer)
# ------------------
if __name__ == "__main__":
	print("Welcome to the local Qwen RAG demo — hit Ctrl+C to exit")
	tokenizer = None
	model = None
	gen = None

	while True:
		try:
			q = input("Question> ").strip()
			if not q:
				continue

			# generate answer
			out = generate_answer(q, top_k=3, model_name=MODEL_NAME)
			print("\n--- ANSWER ---\n")
			print(out["answer"])
			print("\n--- SOURCES ---\n")
			for _id, text, sim in out["contexts"]:
				print(f"[{_id}] (sim={sim:.4f}) {text[:200]}...\n")

		except KeyboardInterrupt:
			print("\nGoodbye")
			break
		except Exception as e:
			print("Error while answering:", e)
			break
