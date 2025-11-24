import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_content_matrix(items_df: pd.DataFrame):
    """
    Cria a matriz de características dos itens usando TF-IDF.
    Junta Título, Categoria, Tags e Sinopse em um único 'sopão' de palavras.
    """
    # Garante que não há valores nulos nos textos
    items_df['tags'] = items_df['tags'].fillna('')
    items_df['synopsis'] = items_df['synopsis'].fillna('')
    items_df['category'] = items_df['category'].fillna('')
    
    # Cria o "sopão" de palavras (Soup) para cada item
    # Damos peso extra para Tags e Categoria repetindo-as
    items_df['soup'] = (
        items_df['title'] + " " + 
        items_df['category'] + " " + items_df['category'] + " " + 
        items_df['tags'] + " " + items_df['tags'] + " " + 
        items_df['synopsis']
    )
    
    # Cria a matriz TF-IDF
    # stop_words='english' remove palavras comuns (the, a, in) se o texto for inglês.
    # Se for português, precisaria de uma lista customizada, mas 'english' não atrapalha muito.
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(items_df['soup'])
    return tfidf_matrix

def get_user_profile(user_id: int, ratings_df: pd.DataFrame, tfidf_matrix):
    """
    Cria o perfil do usuário baseado na média dos vetores dos itens que ele gostou.
    Consideramos "gostou" notas >= 4.
    """
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    # Filtra apenas itens que o usuário avaliou bem (>= 4)
    liked_items = user_ratings[user_ratings['rating'] >= 4]
    
    if liked_items.empty:
        return None
    
    # Pega os índices desses itens na matriz original
    # Assumindo que item_id começa em 1 e os índices da matriz começam em 0:
    liked_indices = liked_items['item_id'].values - 1 
    
    # Filtra índices válidos (caso haja IDs no ratings que não estão no items)
    valid_indices = [i for i in liked_indices if i < tfidf_matrix.shape[0]]
    
    if not valid_indices:
        return None

    # Calcula o vetor médio do usuário (User Profile)
    user_profile = tfidf_matrix[valid_indices].mean(axis=0)
    return user_profile

def get_recommendations(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> list:
    """
    Gera recomendações baseadas na similaridade entre o perfil do usuário e os itens.
    """
    # 1. Prepara a matriz de conteúdo
    tfidf_matrix = create_content_matrix(items_df)
    
    # 2. Cria o perfil do usuário
    user_profile = get_user_profile(user_id, ratings_df, tfidf_matrix)
    
    if user_profile is None:
        return [] # Usuário não tem avaliações positivas suficientes para criar perfil
    
    # 3. Calcula similaridade (Cosseno) entre o perfil do usuário e TODOS os itens
    # user_profile é (1, n_features) e tfidf_matrix é (n_items, n_features)
    cosine_sim = cosine_similarity(user_profile, tfidf_matrix)
    
    # Transforma em array simples
    scores = cosine_sim[0]
    
    # 4. Remove itens que o usuário JÁ viu
    watched_items = ratings_df[ratings_df['user_id'] == user_id]['item_id'].values
    # Ajusta índices (item_id 1 = índice 0)
    watched_indices = watched_items - 1
    
    # Zera o score dos itens já vistos para não recomendar de novo
    for idx in watched_indices:
        if 0 <= idx < len(scores):
            scores[idx] = -1

    # 5. Pega os Top 5
    top_indices = scores.argsort()[::-1][:5]
    
    results = []
    for idx in top_indices:
        if scores[idx] <= 0: continue # Ignora se a similaridade for zero ou item já visto
        
        row = items_df.iloc[idx]
        results.append({
            "item_id": int(row['item_id']),
            "title": row['title'],
            "category": row['category'],
            "score": float(scores[idx]), # Score agora é a similaridade (0 a 1)
            "tags": row['tags'] # Adicionei tags para mostrar no frontend se quiser
        })
        
    return results

def evaluate_accuracy(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Calcula Precision, Recall e F1-Score para um usuário.
    """
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    # Precisamos de itens suficientes para dividir em treino e teste
    positive_ratings = user_ratings[user_ratings['rating'] >= 4]
    if len(positive_ratings) < 2:
        return {"user_id": user_id, "message": "Usuário precisa de pelo menos 2 avaliações positivas (>=4) para o teste."}

    # Divisão Treino/Teste (Garante que o teste tenha pelo menos 1 item positivo)
    test_size = max(1, int(len(positive_ratings) * 0.3))
    test_likes = positive_ratings.sample(test_size, random_state=42)
    
    # O treino é tudo o que NÃO está no teste (inclui notas baixas e as altas restantes)
    train_ratings = user_ratings.drop(test_likes.index)
    
    # Simula o sistema com os dados de treino
    # Importante: passamos train_ratings como se fosse o histórico completo do usuário
    recs = get_recommendations(user_id, items_df, train_ratings)
    
    recommended_ids = {r['item_id'] for r in recs}
    test_liked_ids = set(test_likes['item_id'].values)
    
    # Cálculo das métricas
    hits = len(recommended_ids & test_liked_ids)
    
    # Precision: Dos recomendados, quantos eram bons? (Hits / Total Recomendado)
    precision = hits / len(recs) if len(recs) > 0 else 0
    
    # Recall: Dos que o usuário gostava (teste), quantos eu achei? (Hits / Total no Teste)
    recall = hits / len(test_liked_ids) if len(test_liked_ids) > 0 else 0
    
    # F1-Score: Média harmônica
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "user_id": user_id,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "hits": hits,
        "recommended": list(recommended_ids),
        "test_liked": list(test_liked_ids)
    }

def calculate_overall_accuracy(items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Calcula a média das métricas para todos os usuários.
    """
    unique_users = ratings_df["user_id"].unique()
    precisions = []
    recalls = []
    f1s = []

    for user_id in unique_users:
        result = evaluate_accuracy(user_id, items_df, ratings_df)
        if "precision" in result: # Verifica se calculou com sucesso
            precisions.append(result["precision"])
            recalls.append(result["recall"])
            f1s.append(result["f1_score"])

    if not precisions:
        return {"message": "Nenhum usuário apto para cálculo de métricas."}

    return {
        "mean_precision": sum(precisions) / len(precisions),
        "mean_recall": sum(recalls) / len(recalls),
        "mean_f1": sum(f1s) / len(f1s),
        "users_evaluated": len(precisions)
    }