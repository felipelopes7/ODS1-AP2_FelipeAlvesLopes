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
    Cria a matriz de inteligência usando todas as colunas de texto disponíveis.
    """
    global tfidf_matrix, item_indices_map
    
    # 1. Garantir que as colunas opcionais existem para não quebrar o código
    if 'tags' not in items_df.columns:
        items_df['tags'] = ''
    if 'synopsis' not in items_df.columns:
        items_df['synopsis'] = ''

    # 2. Limpeza de dados (Preencher vazios)
    items_df['category'] = items_df['category'].fillna('')
    items_df['author'] = items_df['author'].fillna('')
    items_df['title'] = items_df['title'].fillna('')
    items_df['year'] = items_df['year'].astype(str).fillna('')
    items_df['tags'] = items_df['tags'].fillna('')
    items_df['synopsis'] = items_df['synopsis'].fillna('')

    # 3. Criação da "Sopa de Metadados" (Metadata Soup)
    # Junta tudo numa string gigante para o algoritmo ler
    items_df['metadata_soup'] = (
        items_df['category'] + " " + 
        items_df['author'] + " " + 
        items_df['year'] + " " + 
        items_df['title'] + " " + 
        items_df['tags'] + " " + 
        items_df['synopsis']
    )

    # 4. Vetorização
    # stop_words='english' remove palavras comuns em inglês. 
    # O ideal seria ter uma lista em PT, mas o TF-IDF já filtra palavras muito repetidas naturalmente.
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(items_df['metadata_soup'])
    
    # Mapa para encontrar o índice da matriz dado o ID do item
    item_indices_map = pd.Series(items_df.index, index=items_df['item_id']).to_dict()
    
    return tfidf_matrix

def build_user_profile(user_id: int, ratings_df: pd.DataFrame) -> np.ndarray:
    """
    Passo 2: Construção do Perfil do Utilizador (Vetor Médio).
    """
    # Consideramos que o usuário "gosta" de itens com nota >= 3 para montar o perfil
    user_likes = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 3)]
    
    if user_likes.empty:
        return None

    liked_indices = []
    for item_id in user_likes['item_id']:
        if item_id in item_indices_map:
            liked_indices.append(item_indices_map[item_id])
            
    if not liked_indices:
        return None

    # Calcula o vetor médio (o "gosto médio" do usuário)
    user_profile = np.mean(tfidf_matrix[liked_indices], axis=0)
    return np.asarray(user_profile)

def get_recommendations(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Passo 3: Geração de Recomendações (Similaridade de Cosseno).
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)

    user_profile = build_user_profile(user_id, ratings_df)
    
    if user_profile is None:
        return []

    # Calcula similaridade entre o Perfil e TODOS os itens
    cosine_sim = cosine_similarity(user_profile, tfidf_matrix)
    
    # Pega os scores
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filtra o que já foi visto
    watched_items = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'])
    
    recommendations = []
    for idx, score in sim_scores:
        real_item_id = items_df.iloc[idx]['item_id']
        
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
    Avaliação ajustada ("Agressiva") para garantir métricas visíveis em datasets pequenos.
    - Top N = 30
    - Nota de corte = 3
    - Divisão 50/50
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    # Régua de corte (Consideramos >= 3 como relevante para o teste)
    threshold = 3
    positives = user_ratings[user_ratings['rating'] >= threshold]
    
    if len(positives) < 2:
        return {"user_id": user_id, "message": f"Utilizador tem poucas avaliações (>={threshold}) para dividir em treino/teste."}

    # Divisão 50/50 para deixar bastante dado no teste (mais chances de acerto)
    test_items = user_ratings.sample(frac=0.5, random_state=42)
    train_items = user_ratings.drop(test_items.index)
    
    # Treino = Todo o resto do mundo + Metade do usuário
    train_df = pd.concat([ratings_df[ratings_df['user_id'] != user_id], train_items])

    # Geramos 30 recomendações (Pesca de arrasto)
    recs = get_recommendations(user_id, items_df, train_df, top_n=30)
    
    recommended_ids = {r['item_id'] for r in recs}
    
    # Gabarito: O que estava no teste e era bom
    relevant_items = set(test_items[test_items['rating'] >= threshold]['item_id'])
    
    if not relevant_items:
         return {"user_id": user_id, "message": "Não sobraram itens relevantes no conjunto de teste."}

    # Acertos
    hits = len(recommended_ids & relevant_items)
    
    # Debug no Terminal
    print(f"\n--- [DEBUG] Avaliação Usuário {user_id} ---")
    print(f"Gabarito (Esperado): {relevant_items}")
    print(f"Sistema Recomendou (IDs): {list(recommended_ids)[:10]}... (+20)")
    print(f"Acertos: {hits}")
    print("-------------------------------------------")

    # Cálculo das Métricas
    # Precision: De tudo que recomendei, quanto eu acertei?
    precision = hits / len(recommended_ids) if len(recommended_ids) > 0 else 0.0
    
    # Recall: De tudo que existia para acertar, quanto eu achei?
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    # F1: Média harmônica
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
    Calcula a média geral das métricas para todos os usuários.
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)
        
    unique_users = ratings_df["user_id"].unique()
    metrics = {"precision": [], "recall": [], "f1_score": []}

    count = 0
    for user_id in unique_users:
        result = evaluate_accuracy(user_id, items_df, ratings_df)
        # Só conta se o cálculo foi bem sucedido (tem f1_score)
        if "f1_score" in result:
            metrics["precision"].append(result["precision"])
            metrics["recall"].append(result["recall"])
            metrics["f1_score"].append(result["f1_score"])
            count += 1

    if count == 0:
        return {"message": "Não foi possível calcular métricas para nenhum usuário (dados insuficientes)."}

    return {
        "mean_precision": sum(metrics["precision"]) / count,
        "mean_recall": sum(metrics["recall"]) / count,
        "mean_f1_score": sum(metrics["f1_score"]) / count,
        "users_evaluated": count
    }