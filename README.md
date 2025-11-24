# *Trabalho ODS1 - AP2: Sistema de Recomendação de Mangás (Content-Based)* #

Segundo Trabalho Prático (AP2) da disciplina Oficina de Desenvolvimento de Sistemas 1.

---

**Equipe**

- Aglison Balieiro Da Silva
- Felipe Alves Lopes
- Leonardo Melo Crispim
- Oziel Bezerra de Lima

---

## Objetivo do Sistema ##

Este projeto implementa uma plataforma de recomendação de mangás baseada em Filtragem por Conteúdo (Content-Based Filtering).

Este sistema analisa as características dos próprios mangás (como gênero, autor, tags e sinopse) para entender o perfil de gosto do utilizador. Se o utilizador avalia bem mangás de "Ação" e "Ninjas", o sistema recomendará outras obras que contenham essas mesmas palavras-chave e descrições similares, independentemente do que outros usuários pensam.

## Cenário de Uso: Recomendação de Mangás

**Por que recomendar mangás?**
O mercado de mangás é vasto, com milhares de títulos lançados anualmente cobrindo dezenas de demografias (Shounen, Seinen, Shoujo) e temas. Leitores frequentemente têm dificuldade em descobrir obras novas que fujam dos títulos mais populares ("mainstream"). Um sistema de recomendação é essencial para conectar leitores a obras de nicho que correspondam aos seus gostos específicos de narrativa e estilo.

**Atributos de Conteúdo Considerados:**
Para realizar a filtragem, utilizamos os seguintes atributos diferenciadores extraídos do nosso catálogo (`items.csv`):
* **Gênero/Categoria:** (ex: Shounen, Seinen, Romance).
* **Tags:** Palavras-chave específicas (ex: "Ninja", "Cyberpunk", "Escolar").
* **Autor:** Para identificar estilo de traço e narrativa.
* **Sinopse:** Descrição textual que fornece o contexto semântico da obra.

## Tecnologias Utilizadas ##

- Linguagem: Python 3.11+

- Backend: FastAPI

- Frontend: Streamlit

- Machine Learning / Processamento de Dados:

- Scikit-learn: Para vetorização de texto (TF-IDF) e cálculo de similaridade (Cosseno).

- Pandas & NumPy: Manipulação de dados e operações vetoriais.

## Instalação e Execução ##

### 1. Configurar o Ambiente Virtual

*Certifique-se de ter o Python 3.11+ instalado.*

Verificar Versão

```bash
python --version
```

Criar Ambiente Virtual

```bash
python -m venv venv
```

Ativat Ambiente Virtual

```bash
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Rodar a Aplicação

*O sistema é dividido em duas partes que devem rodar simultaneamente em terminais diferentes (ambos com o venv ativado).*

**Terminal 1: Backend (API)**

```bash
cd backend
uvicorn app:app --reload
```

*O backend iniciará em http://127.0.0.1:8000 e carregará a matriz de inteligência TF-IDF automaticamente.*

**Terminal 2: Frontend (Interface)**

```bash
cd frontend
streamlit run app_streamlit.py
```

*O navegador abrirá automaticamente com a aplicação.*

## Detalhes da Implementação (O Algoritmo)

A lógica de recomendação reside no arquivo backend/recommender.py e foi construída seguindo os princípios de Processamento de Linguagem Natural (NLP) e Álgebra Linear.

### Passo 1: Preparação e Vetorização (TF-IDF)

O computador não entende texto nativamente, apenas números. Para resolver isso:

- Metadata Soup (Sopa de Metadados): Concatenamos todas as colunas textuais relevantes (Categoria + Autor + Ano + Título + Tags + Sinopse) em uma única string gigante para cada mangá.

- TF-IDF (Term Frequency-Inverse Document Frequency): Aplicamos esta técnica para converter a "sopa" em vetores numéricos.

- O TF-IDF dá pouco peso a palavras muito comuns que não ajudam a diferenciar itens (como "o", "a", "história") e dá muito peso a palavras raras e específicas (como "Alquimia", "Shinigami", "Titã").

O resultado é a tfidf_matrix, uma matriz onde cada linha representa a "impressão digital" matemática de um mangá.

### Passo 2: Construção do Perfil do Utilizador

O sistema precisa traduzir o histórico de notas do usuário em um vetor de preferências:

- Identificamos os itens que o usuário avaliou positivamente (Nota >= 3).

- Recuperamos os vetores TF-IDF desses itens específicos.

- Calculamos a Média Vetorial desses itens. 

O vetor resultante aponta para a "direção média" do gosto do usuário no espaço multidimensional. Por exemplo, se o usuário gosta de "Naruto" e "Bleach", o vetor médio estará posicionado numa região densa de palavras como "Luta", "Poderes" e "Amizade".

### Passo 3: Similaridade de Cosseno

Para gerar a lista final:

- Utilizamos a Similaridade de Cosseno para medir o ângulo entre o Vetor de Perfil do Usuário e os vetores de todos os outros mangás do catálogo.

- Matematicamente, quanto menor o ângulo entre os vetores, mais similar é o conteúdo.

- O sistema ordena os itens pela maior similaridade (score mais próximo de 1.0) e remove aqueles que o usuário já consumiu.

## Métricas e Avaliação de Acurácia

**Metodologia de Teste:**

As avaliações de um utilizador são divididas em Treino (50%) e Teste (50%).

O sistema gera recomendações usando apenas os dados de Treino.

Verifica-se se os itens recomendados aparecem na lista de Teste com avaliações positivas (Nota >= 3).

**Métricas Calculadas:**

Precision: Qual a porcentagem das recomendações geradas que o utilizador realmente gostou?

Recall: Dos itens que o utilizador gosta, quantos o sistema conseguiu encontrar?

F1-Score: Média harmônica entre Precision e Recall, oferecendo um balanço geral da performance.

---

### Estrutura de Arquivos
```bash
/
├── backend/
│   ├── app.py           # API FastAPI e rotas
│   ├── recommender.py   # Lógica do Content-Based Filtering
│   ├── items.csv        # Catálogo de mangás com metadados
│   └── ratings.csv      # Histórico de avaliações
├── frontend/
│   └── app_streamlit.py # Interface gráfica Streamlit
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação
```

### Endpoints da API (Backend)
O backend fornece as seguintes rotas documentadas (Swagger UI disponível em `/docs`):

* `GET /`: Verifica status da API.
* `GET /recomendar/{user_id}`: Retorna recomendações baseadas no perfil vetorial do usuário.
* `GET /avaliar_acuracia/{user_id}`: Calcula Precision, Recall e F1 para um usuário específico.
* `GET /avaliar_acuracia_geral`: Calcula a média de performance de todo o sistema.
