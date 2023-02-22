import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def email_mapper(df):
    ''' Maps the user email to a user_id column and remove the email column
    
    INPUT
        df (dataframe): dataframe containing records of article views (it has 
        3 columns: article_id, article title and user_id)
    OUTPUT:
        email_encoded (list of strings) - newly encoded user_ids
    
    '''

    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = str(cter)
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded


def create_user_item_df(df):
    """ Creates the user-article matrix in dataframe format with users as rows, 
    articles as columns, and values are 1's and 0's (1 if the user
    has read the article regardless of the number of times he/she has 
    read the same article, 0 otherwise)
    
    INPUT:
        df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
        user_item_df - a dataframe with users as rows, articles as columns and 
        values are 1 if the user has seen the article and 0 otherwise 
    """
    # create user_item df with users (ids) as rows, articles (ids) as columns and views as values
    user_item_df = df.groupby(['user_id','article_id'])['title'].count().unstack()
    # replace non-missing values with 1 (any numeric value should be 1) and missing values with 0
    user_item_np = np.matrix(user_item_df)
    np.nan_to_num(user_item_np,0)
    user_item_np[user_item_np>0]=1
    user_item_matrix = pd.DataFrame(user_item_np, 
                                    columns = user_item_df.columns, 
                                    index = user_item_df.index)
    return user_item_matrix


def get_top_article_ids(n, df):
    ''' Returns ids of most popular articles (highest number of views)
    INPUT:
        n - (int) the number of top articles to return
        df (dataframe): dataframe containing records of article views (it has 
        3 columns: article_id, article title and user_id)
    OUTPUT:
        top_articles - (list) A list of the top 'n' article ids
    
    '''
    df_articles_views = df[['article_id','title']].groupby('article_id').count().rename(columns={'title':'views'}).sort_values(by='views', ascending=False).reset_index()
    top_ids = df_articles_views.iloc[:n]['article_id'].tolist()  
    return top_ids # Return the top article ids


def get_top_articles(n, df):
    ''' Returns titles of most popular articles (highest number of views)
    INPUT:
        n - (int) the number of top articles to return
        df (dataframe): dataframe containing records of article views (it has 
        3 columns: article_id, article title and user_id)
    OUTPUT:
        top_articles - (list) A list of the top 'n' article titles 
    
    '''
    top_articles = []
    top_ids = get_top_article_ids(n, df)
    for idx in top_ids:
        top_articles.append(df[df['article_id']==idx]['title'].iloc[0])
    
    return top_articles # Return the top article titles from df (not df_content)


def get_user_articles(user_id, user_item_matrix):
    ''' Provides a list of ids of the articles that have been seen by a user
    INPUT:
        user_id - (int) a user id
        user_item_matrix (dataframe): matrix of users by articles: 
        1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
    
    '''
    article_ids = []
    try:
        user_df = user_item_matrix[user_item_matrix.index==user_id]
        for article_id in user_df.columns: 
            # collect column names where value=1
            if user_df.iloc[0][article_id] == 1:
                article_ids.append(article_id)
    except: # if user_id is new it is not in user_item_matrix
        pass
    return article_ids


def find_similar_articles(article_id, df_content, n_recs = 10):
    ''' Find articles with a title most similar to the input article id
    Article simmilarity is computed by cosine simmilarity between 
    tfidf-vectorized titles. Articles with cosine simmilarity
    equal to zero are excluded
    
    INPUT:
        article_id (int): id of article to find similars
        df_content (dataframe): dataframe of content at IBM Watson Studio
        platform (it has 5 columns: 
                  article_id, 
                  doc_status, 
                  doc_full_name = article title
                  doc_description = article teaser
                  doc_body = article text)
        n_recs (int): number of similar articles to return
        
    OUTPUT:
        similar_ids (list of ints): ids of simmilar articles
        similar_recs (list of strings): titles of simmilar articles
        original_title (string): title of input article for which to find simmilar articles 
    '''
    # vectorize corpus from article titles and store cosine similarity between all articles
    corpus = df_content['doc_full_name'].tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim_article_titles = pd.DataFrame(cosine_sim, 
                                             columns = df_content['article_id'].tolist(), 
                                             index = df_content['article_id'].tolist())
    similar_ids = cosine_sim_article_titles[article_id].sort_values(ascending=False)[1:n_recs+1].index
    # exclude articles with cosine similarity = 0
    similar_ids = [idx for idx in similar_ids if int(idx)>0] 
    if len(similar_ids)>0:
        similar_titles = [df_content[df_content['article_id']==idx]['doc_full_name'].values[0] for idx in similar_ids]
        original_title = df_content[df_content['article_id']==article_id]['doc_full_name'].values[0]
    else:
        similar_titles = 0
        original_title = 0
        
    return original_title, similar_titles,similar_ids


def get_top_sorted_users(user_id, n, dot_prod_df, df):
    ''' Gives a list of users that are most similar to the input user_id 
        and that are the most active
    
    INPUT:
        user_id (int): user for which to find similar users
        n (int): number of users to return
        dot_prod_df (dataframe): a dataframe resulting from the dot product 
        of the user-item matrix with its transpose (contains the similarity 
        of all users with all the other users)
  
        df (pandas dataframe): df containing records of user-article 
        interactions (each row corresponds to 1 user viewing 1 article)
    
            
    OUTPUT:
    neighbors_df: (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the 
                    provided user_id
                    num_interactions - the number of articles viewed by the 
                    user
     
    '''
    # load dot_product_df and get row of user_id from users_similarity dataframe
    user_df = dot_prod_df[dot_prod_df.index==user_id]
    
    # transpose row to have column of similarity and index = neighbour id
    user_df_transpose = user_df.transpose().rename(columns={user_id:'simmilarity'})
    
    # append user activity (total views) 
    user_df_transpose.index = user_df_transpose.index
    user_df_transpose.index.name = 'user_id'
    df_views_users = df[['user_id','title']].groupby(['user_id']).count().rename(columns={'title':'n_views'})
    neighbors_df = user_df_transpose.merge(df_views_users, on='user_id')
    
    # sort 
    neighbors_df.sort_values(by=['simmilarity','n_views'], ascending=False, inplace=True)

    # remove input user
    neighbors_df = neighbors_df[neighbors_df.index != user_id]
    
    return neighbors_df.iloc[:n]


def predicted_articles_ids_svd(user_id, user_item_matrix, k=300):
    ''' Returns articles ids predicted by Single Value Decomposition (SVD) 
    to be liked by user
    INPUT:
        user_id (int): id of user on whom to predict liked articles
        user_item_matrix (2D numpy array): matrix of user-item interactions 
        (unique users for each row and unique articles for each column)
        k (int): number of latent factors to use in SVD
        
    OUTPUT:
        predicted_articles_ids (list of integers): id of articles predicted 
        to be liked for the input user_id
    '''
    # matrix factorization
    u, s, vt = np.linalg.svd(user_item_matrix)
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # obtain predicted user-articles matrix
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
            
    # get list of articles ids predicted to be liked by user
    est_user_row = user_item_est[user_item_matrix.index.get_loc(user_id),:]
    df_pred = pd.DataFrame(est_user_row, index=user_item_matrix.columns)
    predicted_articles_ids = df_pred.loc[df_pred[0]==1].index.tolist()
        
    return predicted_articles_ids


def get_article_names(article_ids, df, df_content):
    ''' Returns the article names associated with a list of article ids
    
    INPUT:
        article_ids - (list) a list of article ids
        df - (pandas dataframe) df containing records of user-article interactions 
        (each row corresponds to 1 user viewing 1 article) 
        df_content (pandas dataframe): df containing title, teaser and body
        text of articles at IBM Watson Studio platform
    
    OUTPUT:
        article_names - (list) a list of article names associated with the 
        list of article ids
    '''

    # then store names
    article_names = []
    for item in article_ids:
        try: # if the article id is in df_content, get name from there
            article_names.append(df_content[df_content['article_id']==item]['doc_full_name'].iloc[0])
        except: # get name from df if not in df_content
            #capitalize string
            capitalized_item = df[df['article_id']==item]['title'].iloc[0].title()
            article_names.append(capitalized_item)            
    return article_names


def final_recommender(user_id, df, df_content, user_item_matrix, dot_prod_df, m=10, k=300):
    ''' Gives a recommendation of articles customized for each type of user 
    (new users, recent users, old users). 
    
    For new users, the most popular articles are pulled and shown ordered by 
    popularity. 
    For recent users, popular articles as well as articles whose titles are 
    similar to those already seen by the user. 
    For old users, in addition to the previous approaches, the system shows
    articles that are also seen by similar users removing those not predicted 
    to be liked by matrix factorization with SVD.
    
    INPUT:
        user_id - (int) a user id
        df - (pandas dataframe) df containing records of user-article interactions 
        (each row corresponds to 1 user viewing 1 article) 
        df_content (pandas dataframe): df containing title, teaser and body
        text of articles at IBM Watson Studio platform        
        user_item_matrix (2D numpy array): matrix of user-item interactions 
        (unique users for each row and unique articles for each column)
        dot_prod_df (dataframe): user similarity (cosine similarity) matrix
        m - (int) the number of recommendations you want for the user
        k (int): number of latent factors to use in SVD
    
    OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title
    
    '''
    # Collect popular artiles. Take more than 10 to ensure 10 are shown, as for 
    # many articles and users there is 0 similarity with other users or articles
    
    popular_articles_ids = get_top_article_ids(20, df)
    popular_articles_names = get_top_articles(20, df)
    
    print("\nStart recommender function")
    ''' new users (no articles read yet) -> popular items
    ''' 
    if user_id not in df['user_id'].unique():
        recs = popular_articles_ids
        rec_names = popular_articles_names
    else:
        ''' common steps for recent and old users
        '''
        recs = []
        # get articles read by user:
        seen_ids = get_user_articles(user_id, user_item_matrix)
        # remove articles not present in df_content (no similarity can be analysed)
        seen_ids_content = [idx for idx in seen_ids if idx in df_content['article_id'].unique()]
        # sort articles seen by number of views
        df_views_articles = df[['article_id','title']].groupby(['article_id']).count().rename(columns={'title':'n_views'})
        #sorted_articles = df_views_articles[df_views_articles.index.isin([int(float(idx)) for idx in seen_ids])].sort_values(by='n_views', ascending=False).index.tolist()           
        sorted_articles = df_views_articles[df_views_articles.index.isin([idx for idx in seen_ids_content])].sort_values(by='n_views', ascending=False).index.tolist() 
        ''' Recent users (less than 5 articles read) -> up to 80% similar articles, 
        at least 20% popular items 
        5 is an arbitrary threshold, could be optimized, 
        but visual exploration suggest it is ok
        '''
        if len(seen_ids) < 6:
            print("start recent user")
            # collect similar articles 
            for article in sorted_articles:
                # find similar articles (4 for each seen article, to add variety)
                similar_ids = find_similar_articles(article, df_content, n_recs = 4)[2]
                # store and remove those already seen by user
                recs.extend(similar_ids)
                recs = list(set(recs) - set(seen_ids))        
                # if number of collected recommendations exceeds m, stop loop
                if len(recs) > int(0.8 * m) -1 :
                    recs = recs[:int(0.8 * m)]
                    break       
            print("finished similar articles, collected {} articles".format(len(recs)))
        else:
            ''' old users (more than 5 articles read) -> up to 50% similar articles, 
            at least 50% articles from simmilar users, 
            added SVD-prediction of articles based on other users, 
            and until 100% popular items '''
            
            print("start old user")
            # similar articles (content similarity)
            for article in sorted_articles:
                # find similar articles (2 for each seen article, to add variety)
                similar_ids = find_similar_articles(article, df_content, n_recs = 2)[2]
                # store and remove those already seen by user
                recs.extend(similar_ids)
                recs = list(set(recs) - set(seen_ids))        
                # if number of collected recommendations exceeds m, stop loop
                if len(recs) > int(0.7 * m) -1 :
                    recs = recs[:(int(0.7 * m) -1)]
                    break 
            print("finished similar articles, collected {} articles".format(len(recs)))
            
            # only take predicted articles if the user is very active
            if len(seen_ids) > 100:
                print("start user-item predicted filtering")
                # compute SVD-predicted user-item matrix
                predicted_articles = predicted_articles_ids_svd(user_id, user_item_matrix, k=k)
                print("There are {} predicted articles".format(len(predicted_articles)))
                #sort those predicted articles by max to min popularity and store them in recs
                sorted_predicted_articles = df_views_articles[df_views_articles.index.isin(predicted_articles)].sort_values(by='n_views', ascending=False).index.tolist()
                recs.extend(sorted_predicted_articles)
                #recs.extend(predicted_articles)
                recs = list(set(recs) - set(seen_ids))
                if len(recs) > m-2:
                    recs = recs[:(m-2)]   
                print("finished predicted articles, collected {} articles".format(len(recs)))
            
            print("start similar users")
            # articles from similar users
            # bring 10 most similar users
            similar_users = get_top_sorted_users(user_id, 10, dot_prod_df, df).index.tolist()
            for user in similar_users:
                #print("Started user {}".format(user))
                # For each user get his/her top 2 articles
                new_ids = get_user_articles(user, user_item_matrix)[:2] 
                sorted_articles_user = df_views_articles[df_views_articles.index.isin(new_ids)].sort_values(by='n_views', ascending=False).index.tolist()
                # save ids in recs
                recs.extend(sorted_articles_user)
                # remove articles already seen by input user
                recs = list(set(recs) - set(seen_ids))
                # if number of collected recommendations exceeds m, stop loop
                if len(recs) > m-1:
                    recs = recs[:(m-1)]            
                    break
            print("finished similar users, collected {} articles".format(len(recs)))
        # Finally, for both old and recent users, after collecting similar articles 
        # (based on content or user similarity), add popular articles to recs
        if len(recs)< m:
            for idx in popular_articles_ids:
                recs.append(idx)
                recs = list(set(recs) - set(seen_ids))
                if len(recs) > m-1:
                    print("finished popular articles, collected {} articles".format(len(recs)))
                    break

    # get only the number of recs requested by m as parameter:
    recs = recs[:m]
    rec_names = get_article_names(recs, df, df_content)   
    print("Finished recommender function\n")
    return recs, rec_names
