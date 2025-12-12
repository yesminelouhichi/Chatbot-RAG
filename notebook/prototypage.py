# Faire les importations nécessaires
import google.generativeai as genai
import psycopg
from psycopg import Cursor
import os
import glob

# Déclarer les variables nécessaires
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
# Backwards-compatible single-file variable (not used by default anymore)
conversation_file_path = os.path.join(data_dir, "017_00000012.txt")

# Initialiser le client Gemini
genai.configure(api_key="AIzaSyBA46qJcksx81VvKVdaV_ZBQqQNyNs6w80")

db_connection_str = "dbname=rag_chatbot user=postgres password=votremotdepasse host=localhost port=5432"

def create_conversation_list(file_path: str) -> list[str]:
    """Lit le fichier avec le bon encodage et filtre les lignes"""
    # 🔥 FIX 1: essayer UTF-8 puis fallback sur latin-1 / cp1252 si nécessaire
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except UnicodeDecodeError:
        print(f"⚠️  Encodage UTF-8 invalide pour {file_path} — réessayage avec cp1252 (latin-1)")
        with open(file_path, "r", encoding="cp1252", errors="replace") as file:
            text = file.read()

    # maintenant traiter le texte (commun pour UTF-8 et cp1252)
    text_list = text.split("\n")
    # 🔥 FIX 2: Filtrer les lignes vides avec .strip()
    filtered_list = [
        chaine.removeprefix("     ")
        for chaine in text_list
        if not chaine.startswith("<") and chaine.strip()  # Enlever les vides
    ]
    print(f"✓ {len(filtered_list)} lignes extraites")
    return filtered_list


def calculate_embeddings(corpus: str) -> list[float]:
    """Calcule les embeddings avec Gemini"""
    # 🔥 FIX 3: Vérifier que le corpus n'est pas vide
    if not corpus or not corpus.strip():
        raise ValueError("Le corpus ne peut pas être vide")
    
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=corpus,
        task_type="retrieval_document"
    )
    return response["embedding"]


def save_embedding(corpus: str, embedding: list[float], cursor: Cursor) -> None:
    """Sauvegarde le corpus et son embedding"""
    cursor.execute(
        '''INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)''',
        (corpus, embedding)
    )


def similar_corpus(input_corpus: str, db_connection_str: str, top_k: int = 5) -> list[tuple]:
    """
    Fonction qui prend en entrée un texte et renvoie les textes similaires 
    dans la base de données
    """
    # Calculer l'embedding de la requête
    query_embedding = calculate_embeddings(input_corpus)
    
    with psycopg.connect(db_connection_str) as conn:
        with conn.cursor() as cur:
            # Recherche par similarité cosine avec pgvector
            cur.execute("""
                SELECT id, corpus, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            return cur.fetchall()


# Programme principal
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 CRÉATION DE LA BASE D'EMBEDDINGS")
    print("=" * 70)
    
    with psycopg.connect(db_connection_str) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Supprimer la table si elle existe
            cur.execute("""DROP TABLE IF EXISTS embeddings""")
            print("✓ Table existante supprimée")
            
            # 🔥 FIX 4: CRÉER L'EXTENSION PGVECTOR 
            cur.execute("""CREATE EXTENSION IF NOT EXISTS vector""")
            print("✓ Extension pgvector créée")
            
            # 🔥 FIX 5: Utiliser VECTOR(768) au lieu de FLOAT8[]
            # Gemini text-embedding-004 produit des vecteurs de 768 dimensions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY, 
                    corpus TEXT,
                    embedding VECTOR(768)
                )
            """)
            print("✓ Table embeddings créée avec VECTOR(768)")
            
            # Créer un index pour accélérer les recherches de similarité
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_embedding_idx 
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            print("✓ Index de recherche créé")
            
            # Charger les corpus
            print("\n" + "=" * 70)
            print("📂 CHARGEMENT DES DONNÉES")
            print("=" * 70)
            # Parcourir tous les fichiers .txt dans le dossier data
            text_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
            if not text_files:
                print(f"⚠️  Aucun fichier .txt trouvé dans le dossier {data_dir}")

            # Traiter les embeddings (par ligne / par entrée) pour chaque fichier
            print("\n" + "=" * 70)
            print("⚙️  TRAITEMENT DES EMBEDDINGS")
            print("=" * 70)
            
            success_count = 0
            error_count = 0
            
            total_files = len(text_files)
            file_idx = 0
            for file_idx, file_path in enumerate(text_files, 1):
                print(f"\n🔸 Traitement du fichier [{file_idx}/{total_files}] : {os.path.basename(file_path)}")
                corpus_list = create_conversation_list(file_path=file_path)

                # Traiter les embeddings pour chaque entrée du fichier
                for i, corpus in enumerate(corpus_list, 1):
                    try:
                        embedding = calculate_embeddings(corpus)
                        save_embedding(corpus=corpus, embedding=embedding, cursor=cur)
                        success_count += 1
                    
                        # Afficher un aperçu
                        preview = corpus[:50] + "..." if len(corpus) > 50 else corpus
                        print(f"✓ [{i}/{len(corpus_list)}] {preview}")
                    
                    except Exception as e:
                        error_count += 1
                        print(f"✗ [{i}/{len(corpus_list)}] ERREUR: {e}")
            
            conn.commit()
            
            # Résumé
            print("\n" + "=" * 70)
            print("📊 RÉSUMÉ")
            print("=" * 70)
            print(f"✓ Succès: {success_count}")
            print(f"✗ Erreurs: {error_count}")
            print(f"📦 Total sauvegardé: {success_count}")
            
            # Test de recherche
            if success_count > 0:
                print("\n" + "=" * 70)
                print("🔍 TEST DE RECHERCHE")
                print("=" * 70)
                
                test_query = "stage anglais espagnol"
                print(f"Requête: '{test_query}'")
                
                try:
                    results = similar_corpus(test_query, db_connection_str, top_k=3)
                    print(f"\n📌 Top 3 résultats:")
                    for doc_id, corpus, similarity in results:
                        preview = corpus[:60] + "..." if len(corpus) > 60 else corpus
                        print(f"  [Score: {similarity:.4f}] {preview}")
                except Exception as e:
                    print(f"❌ Erreur: {e}")
            
            print("\n" + "=" * 70)
            print("✅ TERMINÉ AVEC SUCCÈS!")
            print("=" * 70)
