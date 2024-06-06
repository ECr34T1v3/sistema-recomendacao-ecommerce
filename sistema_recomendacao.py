import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Simulação de dados de compras
data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'product_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'product_name': [
        'Laptop', 'Smartphone', 'Tablet', 'Smartwatch', 'Camera',
        'Headphones', 'Printer', 'Monitor', 'Keyboard', 'Mouse'
    ],
    'product_category': [
        'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics',
        'Accessories', 'Electronics', 'Electronics', 'Accessories', 'Accessories'
    ]
}

# Criação do DataFrame
df = pd.DataFrame(data)

# Visualização dos dados
print("Dados de Compras:\n", df)

# Criação da matriz de contagem
count = CountVectorizer()
count_matrix = count.fit_transform(df['product_category'])

# Cálculo da similaridade de cosseno
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Função de recomendação
def get_recommendations(product_name, cosine_sim=cosine_sim):
    idx = df.index[df['product_name'] == product_name][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    product_indices = [i[0] for i in sim_scores]
    return df['product_name'].iloc[product_indices]

# Exemplo de recomendação
product_to_recommend = 'Laptop'
recommendations = get_recommendations(product_to_recommend)

print(f"\nProdutos recomendados para '{product_to_recommend}':")
for product in recommendations:
    print(product)

# Salvando o DataFrame para uso futuro
df.to_csv('ecommerce_data.csv', index=False)
