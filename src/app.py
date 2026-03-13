import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Matcher", layout="wide")

@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender=load_recommender()

st.sidebar.header("Filter Preferences")
max_time=st.sidebar.slider("Max Runtime (Mins)", 60, 240, 180)
sim_weight=st.sidebar.slider("Similarity Weight", 0.0, 1.0, 0.7)
rating_weight=1.0-sim_weight

st.title("Movie Matcher")
st.markdown("Enter a movie you like, I will recommend you a similar match!")
user_input=st.text_input("Enter the title of a movie:")

if user_input:
    with st.spinner("Loading recommender..."):
        result=recommender.recommend(user_input, n=6, max_runtime=max_time, similarity_weight=sim_weight, rating_weight=rating_weight)
    if isinstance(result, str):
        st.warning(result)
    else:
        st.subheader('I think you would like: ')
        cols=st.columns(3)
        for i, (idx, row) in enumerate(result.iterrows()):
            with cols[i%3]:
                st.info(f'**{row['Movie_Title']}**')
                st.write(f'Rating: {row['Rating']}/10')
                st.write(f'{row['Runtime(Mins)']} mins')




#jjbjj