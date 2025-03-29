import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import streamlit as st
import matplotlib.pyplot as plt

books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

ratings = ratings[ratings['Book-Rating'] > 0]
book_counts = ratings['ISBN'].value_counts()
popular_books = book_counts[book_counts >= 50].index
filtered_ratings = ratings[ratings['ISBN'].isin(popular_books)]

user_counts = filtered_ratings['User-ID'].value_counts()
active_users = user_counts[user_counts >= 10].index
filtered_ratings = filtered_ratings[filtered_ratings['User-ID'].isin(active_users)]

user_item_matrix = filtered_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
user_item_sparse = csr_matrix(user_item_matrix.values)

n_components = 20
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item_sparse)
item_factors = svd.components_

user_knn = NearestNeighbors(metric='cosine', algorithm='brute')
user_knn.fit(user_factors)

books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
age_groups = pd.cut(users['Age'], bins=[0, 18, 25, 35, 50, 100], labels=['<18', '18-25', '26-35', '36-50', '50+'])
users['Age-Group'] = age_groups
merged_data = filtered_ratings.merge(books, on='ISBN').merge(users[['User-ID', 'Location', 'Age-Group']], on='User-ID')


def get_user_info(user_id):
    user_info = users[users['User-ID'] == user_id]
    if not user_info.empty:
        age_group = user_info['Age-Group'].values[0]
        location = user_info['Location'].values[0]
        return age_group, location
    return None, None

def recommend_books(user_id=None, n_recommendations=10, age_group=None, location=None, read_books=None):
    recommendations = pd.DataFrame(columns=['Book-Title', 'Image-URL-L'])

    if user_id in user_item_matrix.index:
        user_index = user_item_matrix.index.get_loc(user_id)
        distances, indices = user_knn.kneighbors(user_factors[user_index].reshape(1, -1),
                                                 n_neighbors=n_recommendations + 1)

        similar_user_ids = [user_item_matrix.index[i] for i in indices.flatten() if i != user_index]
        similar_user_ratings = ratings[ratings['User-ID'].isin(similar_user_ids)]

        popular_books = similar_user_ratings['ISBN'].value_counts().index[:n_recommendations]
        recommendations_cf = merged_data[merged_data['ISBN'].isin(popular_books)][
            ['Book-Title', 'Image-URL-L']].drop_duplicates()
        recommendations = pd.concat([recommendations, recommendations_cf]).drop_duplicates().head(n_recommendations)

    if age_group:
        similar_age_users = merged_data[merged_data['Age-Group'] == age_group]['User-ID'].unique()
        age_books = merged_data[merged_data['User-ID'].isin(similar_age_users)]['ISBN'].value_counts().head(
            n_recommendations // 2).index
        age_recommendations = merged_data[merged_data['ISBN'].isin(age_books)][
            ['Book-Title', 'Image-URL-L']].drop_duplicates()
        recommendations = pd.concat([recommendations, age_recommendations]).drop_duplicates().head(n_recommendations)

    if location:
        location_users = merged_data[merged_data['Location'].str.contains(location, case=False, na=False)][
            'User-ID'].unique()
        loc_books = merged_data[merged_data['User-ID'].isin(location_users)]['ISBN'].value_counts().head(
            n_recommendations // 2).index
        location_recommendations = merged_data[merged_data['ISBN'].isin(loc_books)][
            ['Book-Title', 'Image-URL-L']].drop_duplicates()
        recommendations = pd.concat([recommendations, location_recommendations]).drop_duplicates().head(
            n_recommendations)

    if recommendations.empty:
        popular_books = merged_data['ISBN'].value_counts().index[:n_recommendations]
        recommendations = merged_data[merged_data['ISBN'].isin(popular_books)][
            ['Book-Title', 'Image-URL-L']].drop_duplicates()

    return recommendations


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def evaluate_model():
    sample_ratings = ratings.sample(frac=0.1, random_state=42)

    train_data, test_data = train_test_split(sample_ratings, test_size=0.2, random_state=42)

    train_matrix = train_data.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
    train_sparse_matrix = csr_matrix(train_matrix.values)

    svd = TruncatedSVD(n_components=20, random_state=42)
    user_factors = svd.fit_transform(train_sparse_matrix)
    item_factors = svd.components_

    rmse_values = []
    mae_values = []
    precision_at_k = []
    total_books = set(train_matrix.columns)
    recommended_books = set()
    personalization_scores = []

    predicted_ratings = []
    for _, row in test_data.iterrows():
        user_id, isbn, actual_rating = row['User-ID'], row['ISBN'], row['Book-Rating']

        if user_id in train_matrix.index and isbn in train_matrix.columns:
            user_index = train_matrix.index.get_loc(user_id)
            item_index = train_matrix.columns.get_loc(isbn)

            predicted_rating = np.dot(user_factors[user_index], item_factors[:, item_index])
            rmse_values.append((predicted_rating - actual_rating) ** 2)
            mae_values.append(abs(predicted_rating - actual_rating))
            predicted_ratings.append(predicted_rating)
        else:
            predicted_ratings.append(np.nan)  # Để giá trị NaN nếu không có dữ liệu để dự đoán

    rmse = sqrt(np.mean(rmse_values))
    mae = np.mean(mae_values)

    for user_id in train_matrix.index:
        user_index = train_matrix.index.get_loc(user_id)
        user_vector = user_factors[user_index]


        scores = np.dot(user_vector, item_factors)
        top_k_indices = np.argsort(scores)[-10:]
        top_k_isbns = train_matrix.columns[top_k_indices]

        relevant_items = test_data[(test_data['User-ID'] == user_id) &
                                   (test_data['ISBN'].isin(top_k_isbns))]
        precision = len(relevant_items) / 10
        precision_at_k.append(precision)

        # Coverage: Đếm tổng số sách được gợi ý ít nhất một lần
        recommended_books.update(top_k_isbns)

    # Coverage = Tỷ lệ sách được gợi ý trên tổng số sách
    coverage = len(recommended_books) / len(total_books)

    # Personalization: Đo lường sự đa dạng trong gợi ý giữa các người dùng
    all_top_k_sets = []
    for user_id in train_matrix.index:
        user_index = train_matrix.index.get_loc(user_id)
        user_vector = user_factors[user_index]

        # Lấy top 10 sách gợi ý cho người dùng này
        scores = np.dot(user_vector, item_factors)
        top_k_indices = np.argsort(scores)[-10:]
        all_top_k_sets.append(set(top_k_indices))

    # Tính personalization bằng cách tính trung bình Jaccard distance giữa các top 10 sets
    for i in range(len(all_top_k_sets)):
        for j in range(i + 1, len(all_top_k_sets)):
            intersection = len(all_top_k_sets[i].intersection(all_top_k_sets[j]))
            union = len(all_top_k_sets[i].union(all_top_k_sets[j]))
            jaccard_similarity = intersection / union
            personalization_scores.append(1 - jaccard_similarity)  # Jaccard distance

    personalization = np.mean(personalization_scores)

    # Vẽ biểu đồ phân bố lỗi
    fig, ax = plt.subplots()
    errors = np.array(rmse_values) ** 0.5 - np.array(mae_values)
    ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    ax.set_title("Error Distribution")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")

    # Kết quả đánh giá
    results = {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': np.mean(precision_at_k),
        'Coverage': coverage,
        'Personalization': personalization
    }

    # Hiển thị kết quả và biểu đồ
    return results, fig


st.title("Enhanced Book Recommender System with Model Evaluation")

user_id_input = st.text_input("Enter User ID (if available):")

age_group, location = None, None

if user_id_input:
    try:
        user_id = int(user_id_input)
        age_group, location = get_user_info(user_id)
    except ValueError:
        st.error("Please enter a valid User ID.")

age_group_input = st.selectbox("Select Age Group:", options=['<18', '18-25', '26-35', '36-50', '50+'],
                               index=['<18', '18-25', '26-35', '36-50', '50+'].index(age_group) if age_group else 0)
location_input = st.text_input("Enter Location (optional):", value=location if location else "")

read_books_input = st.text_input("Enter books you've read (separated by commas):")

if st.button("Show Recommendations"):
    recommendations = recommend_books(
        user_id=int(user_id_input) if user_id_input else None,
        age_group=age_group_input,
        location=location_input,
        read_books=read_books_input
    )

    st.write("Recommended Books:")
    for _, row in recommendations.iterrows():
        st.write(f"**{row['Book-Title']}**")
        if pd.notna(row['Image-URL-L']):
            st.image(row['Image-URL-L'], width=150)

if st.button("Evaluate Model"):
    rmse, mae, fig = evaluate_model()
    st.write(f"Model Evaluation Metrics:\n - RMSE: {rmse}\n - MAE: {mae}")
    st.pyplot(fig)
