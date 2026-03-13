import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, utils

class MovieRecommender:

    def __init__(self, csv_file='src/movies.csv'):
        self.movies=pd.read_csv(csv_file)

        self.movies["Director"]=self.movies["Director"].fillna("")
        self.movies["Actors"]=self.movies["Actors"].fillna("")
        self.movies["main_genre"]=self.movies["main_genre"].fillna("")
        self.movies["side_genre"]=self.movies["side_genre"].fillna("")
        self.movies["Runtime(Mins)"]=pd.to_numeric(self.movies["Runtime(Mins)"])

        self.movies["features"]=(
            self.movies["Director"]+" "+
            self.movies["Actors"]+" "+
            self.movies["main_genre"]+" "+
            self.movies["side_genre"]
        )
        self.movies["features"]=self.movies["features"].str.replace(",", "")
        vectorizer=TfidfVectorizer(stop_words='english')
        self.tfidf_matrix=vectorizer.fit_transform(self.movies["features"])


    def recommend(self,title,n=5,max_runtime=None, similarity_weight=0.7, rating_weight=0.3):
        all_titles=self.movies["Movie_Title"].tolist()
        best_match=process.extractOne(title, all_titles,processor=utils.default_process)

        if best_match[1]<60:
            return f"Movie not found. Did you mean: "+ best_match[0] + "?"
        idx=best_match[2]

        movie_vector=self.tfidf_matrix[idx]
        similarities=cosine_similarity(movie_vector, self.tfidf_matrix).flatten()

        avg_rating=self.movies["Rating"].mean()
        normalized_ratings=self.movies["Rating"].fillna(avg_rating)/10.0

        scores=(similarities*similarity_weight)+(normalized_ratings*rating_weight)

        if max_runtime is not None:
            time_mask=self.movies["Runtime(Mins)"]<=max_runtime
            scores=scores*time_mask- (~time_mask)

        all_sorted_indices=scores.argsort()[::-1]
        recommended_indices=[i for i in all_sorted_indices if i!=idx]


        return self.movies.iloc[recommended_indices[:n]][["Movie_Title", "Rating", "Runtime(Mins)"]]