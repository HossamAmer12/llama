import json, os, networkx as nx, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # noqa: F401
from sentence_transformers import SentenceTransformer
import anthropic

client = anthropic.Anthropic()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── 1. Your documents ─────────────────────────────────────────────
docs = [
    "Alice is a senior engineer at Acme Corp. She leads the platform team.",
    "Acme Corp acquired Startup X in 2023. Startup X built a caching library.",
    "Bob reports to Alice. Bob is working on integrating Startup X's caching library.",
]

# ── 2. Embed chunks ───────────────────────────────────────────────
embeddings = embedder.encode(docs)

print("Embeddings shape:", embeddings.shape)  # should be (3, 384) for all-MiniLM-L6-v2

# ── 3. Extract entities with Claude ──────────────────────────────
def extract_triples(text):
    resp = client.messages.create(
        model="claude-haiku-4-5",   # fast + cheap for extraction
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Extract entities and relationships from this text.
Return ONLY a JSON list like: [{{"source": "Alice", "relation": "WORKS_AT", "target": "Acme Corp"}}]
Text: {text}"""
        }]
    )
    raw = resp.content[0].text.strip()
    # strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── 4. Build knowledge graph ──────────────────────────────────────
G = nx.DiGraph()
chunk_entities = []

for i, doc in enumerate(docs):
    triples = extract_triples(doc)
    entities = set()
    for t in triples:
        src, rel, tgt = t["source"], t["relation"], t["target"]
        G.add_node(src)
        G.add_node(tgt)
        G.add_edge(src, tgt, relation=rel)
        entities.update([src, tgt])
    
    # Store the entities found in 
    # this chunk for later retrieval
    chunk_entities.append(entities)
    print(f"Chunk {i}: {triples}")  # so you can see what was extracted

# ── 5. Query function ─────────────────────────────────────────────
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def ask(question, top_k=2, hops=2):
    # Vector retrieval
    q_emb = embedder.encode([question])[0]
    sims = [cosine_sim(q_emb, e) for e in embeddings]
    top_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:top_k]

    # Seed nodes from top chunks
    seeds = set()
    for idx in top_idx:
        seeds.update(chunk_entities[idx])

    # BFS graph traversal
    edges_found = []
    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                rel = G.edges[node, neighbor]["relation"]
                edges_found.append(f"{node} --[{rel}]--> {neighbor}")
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    chunk_context = "\n".join([docs[i] for i in top_idx])
    graph_context = "\n".join(edges_found) or "None"

    print("\n── SEEDS ──────────────────────────────")
    print(seeds)
    print("\n── CHUNK CONTEXT (raw text sent to Claude) ──")
    print(chunk_context)
    print("\n── GRAPH CONTEXT (edges found via BFS) ──")
    print(graph_context)
    print("───────────────────────────────────────\n")

    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using the context below.

Text chunks:
{chunk_context}

Knowledge graph:
{graph_context}

Question: {question}"""
        }]
    )
    return resp.content[0].text

# ── 6. Basic RAG (no graph) — for comparison ─────────────────────
def basic_rag(question, top_k=2):
    q_emb = embedder.encode([question])[0]
    sims = [cosine_sim(q_emb, e) for e in embeddings]
    top_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:top_k]
    chunk_context = "\n".join([docs[i] for i in top_idx])

    print("\n── BASIC RAG: chunks retrieved ──")
    for i in top_idx:
        print(f"  [{i}] (sim={sims[i]:.2f}): {docs[i]}")

    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using only the text below.

Text chunks:
{chunk_context}

Question: {question}"""
        }]
    )
    return resp.content[0].text

# ── 7. Visualize the knowledge graph ──────────────────────────────
def visualize_graph(G, question=None, seeds=None):
    """Draw the knowledge graph with labeled edges."""
    _, ax = plt.subplots(figsize=(14, 8))

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Color nodes: orange if they are seed nodes (matched by query), else steelblue
    node_colors = []
    for node in G.nodes():
        if seeds and node in seeds:
            node_colors.append("#f4a261")   # orange = query-relevant
        else:
            node_colors.append("#4a90d9")   # blue = regular

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#555", arrows=True,
                           arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax)

    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7,
                                 font_color="#c0392b", ax=ax)

    legend = [
        mpatches.Patch(color="#4a90d9", label="Entity"),
        mpatches.Patch(color="#f4a261", label="Query-matched entity"),
    ]
    ax.legend(handles=legend, loc="upper left")
    ax.set_title(f"Knowledge Graph\nQuery: {question}" if question else "Knowledge Graph", pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("graphrag_viz.png", dpi=150)
    print("Graph saved to graphrag_viz.png")
    plt.show()

# ── 8. Compare basic RAG vs GraphRAG ─────────────────────────────
# This question requires connecting Bob → Alice → Acme Corp across chunks.
# Basic RAG can only use what's in the retrieved text.
# GraphRAG also walks the graph edges, bridging the gap.
q = "What is Bob working on, and who does his manager work for?"
print("\n" + "="*55)
print("Q:", q)
print("="*55)
print("\n[BASIC RAG]")
print(basic_rag(q))
print("\n[GRAPH RAG]")
print(ask(q))

# Show which seeds the query found, then visualize
q_emb = embedder.encode([q])[0]
sims = [cosine_sim(q_emb, e) for e in embeddings]
top_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:2]
seeds = set()
for idx in top_idx:
    seeds.update(chunk_entities[idx])
print("\nQuery-matched seeds (orange nodes):", seeds)

visualize_graph(G, question=q, seeds=seeds)