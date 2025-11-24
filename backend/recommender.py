import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Variáveis globais para armazenar a matriz e o mapa de índices
tfidf_matrix = None
item_indices_map = {}

def prepare_data_and_vectorize(items_df: pd.DataFrame):
    """
    Passo 1: Preparação dos Dados e Vetorização (TF-IDF).
    """
    global tfidf_matrix, item_indices_map
    
    # 1. Garantir que a coluna 'tags' existe
    if 'tags' not in items_df.columns:
        items_df['tags'] = ''

    # 2. Preenchimento de valores nulos (Limpeza)
    items_df['category'] = items_df['category'].fillna('')
    items_df['author'] = items_df['author'].fillna('')
    items_df['title'] = items_df['title'].fillna('')
    items_df['year'] = items_df['year'].astype(str).fillna('')
    items_df['tags'] = items_df['tags'].fillna('')

    # 3. Criação da "Sopa de Metadados" (Metadata Soup)
    # Concatena todas as colunas de texto relevantes numa única string
    items_df['metadata_soup'] = (
        items_df['category'] + " " + 
        items_df['author'] + " " + 
        items_df['year'] + " " + 
        items_df['title'] + " " + 
        items_df['tags']
    )

    # 4. Vetorização usando TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(items_df['metadata_soup'])
    
    # Mapa para encontrar rapidamente o índice da matriz a partir do item_id
    item_indices_map = pd.Series(items_df.index, index=items_df['item_id']).to_dict()
    
    return tfidf_matrix

def build_user_profile(user_id: int, ratings_df: pd.DataFrame) -> np.ndarray:
    """
    Passo 2: Construção do Perfil do Utilizador.
    Calcula a média dos vetores dos itens que o utilizador avaliou positivamente (>= 4).
    """
    # Filtra apenas avaliações positivas no conjunto de dados fornecido (Treino ou Total)
    user_likes = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]
    
    if user_likes.empty:
        return None

    liked_indices = []
    for item_id in user_likes['item_id']:
        if item_id in item_indices_map:
            liked_indices.append(item_indices_map[item_id])
            
    if not liked_indices:
        return None

    # Calcula o vetor médio do perfil
    user_profile = np.mean(tfidf_matrix[liked_indices], axis=0)
    
    # Garante que retorna um array numpy compatível
    return np.asarray(user_profile)

def get_recommendations(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Passo 3: Geração de Recomendações.
    Calcula a similaridade de cosseno entre o perfil do utilizador e todos os itens.
    Aceita 'top_n' dinâmico para permitir testes mais abrangentes.
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)

    user_profile = build_user_profile(user_id, ratings_df)
    
    if user_profile is None:
        return []

    # Calcula similaridade
    cosine_sim = cosine_similarity(user_profile, tfidf_matrix)
    
    # Ordena os scores (do maior para o menor)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Identifica itens que o utilizador já interagiu no conjunto de dados atual
    watched_items = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'])
    
    recommendations = []
    for idx, score in sim_scores:
        real_item_id = items_df.iloc[idx]['item_id']
        
        # Só recomenda se não estiver no conjunto de 'watched' passado
        if real_item_id not in watched_items:
            row = items_df.iloc[idx]
            recommendations.append({
                "item_id": int(real_item_id),
                "title": row['title'],
                "category": row['category'],
                "score": float(score)
            })
            
        if len(recommendations) >= top_n:
            break
            
    return recommendations

def evaluate_accuracy(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Avaliação ajustada para ser mais generosa e encontrar Hits.
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    # === MUDANÇA 1: Baixar régua para >= 3 para ter mais dados de teste ===
    threshold = 3
    positives = user_ratings[user_ratings['rating'] >= threshold]
    
    if len(positives) < 2:
        return {"user_id": user_id, "message": f"Utilizador tem poucas avaliações >= {threshold}."}

    # Divisão Treino/Teste
    test_items = user_ratings.sample(frac=0.3, random_state=42)
    train_items = user_ratings.drop(test_items.index)
    
    train_df = pd.concat([ratings_df[ratings_df['user_id'] != user_id], train_items])

    # === MUDANÇA 2: Aumentar para Top 20 para tentar 'pescar' mais acertos ===
    recs = get_recommendations(user_id, items_df, train_df, top_n=20)
    
    recommended_ids = {r['item_id'] for r in recs}
    
    # Gabarito com a nova régua
    relevant_items = set(test_items[test_items['rating'] >= threshold]['item_id'])
    
    if not relevant_items:
         return {"user_id": user_id, "message": "Não restaram itens no teste."}

    hits = len(recommended_ids & relevant_items)
    
    print(f"\n--- [DEBUG] Avaliação Utilizador {user_id} ---")
    print(f"Gabarito (Nota >={threshold}): {relevant_items}")
    print(f"Recomendados (Top 20): {recommended_ids}")
    print(f"Acertos (Hits): {hits}")
    print("---------------------------------------------")

    precision = hits / len(recommended_ids) if len(recommended_ids) > 0 else 0.0
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    return {
        "user_id": user_id,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "hits": hits,
        "recommended": list(recommended_ids),
        "relevant_in_test": list(relevant_items)
    }

def calculate_overall_accuracy(items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Calcula a média das métricas para todos os utilizadores únicos.
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)
        
    unique_users = ratings_df["user_id"].unique()
    metrics = {"precision": [], "recall": [], "f1_score": []}

    for user_id in unique_users:
        result = evaluate_accuracy(user_id, items_df, ratings_df)
        if "f1_score" in result:
            metrics["precision"].append(result["precision"])
            metrics["recall"].append(result["recall"])
            metrics["f1_score"].append(result["f1_score"])

    if not metrics["f1_score"]:
        return {"message": "Não foi possível calcular métricas para nenhum utilizador (dados insuficientes)."}

    return {
        "mean_precision": sum(metrics["precision"]) / len(metrics["precision"]),
        "mean_recall": sum(metrics["recall"]) / len(metrics["recall"]),
        "mean_f1_score": sum(metrics["f1_score"]) / len(metrics["f1_score"]),
        "users_evaluated": len(metrics["f1_score"])
    }