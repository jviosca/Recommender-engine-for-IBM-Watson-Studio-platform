import pandas as pd
import numpy as np
import recommender_functions as f

class Recommender():
    ''' Makes recommendations of articles from IBM Watson Studio platform using 
    a combination of ranked-based recommendations, 
    content-based recommendations and collaborative-filtering.
    
    For new users, the recommender shows popular articles.
    
    Once a user starts viewing articles, those articles are recorded to 
    recommend articles which text title is simmilar.
    
    Once a user has seen more than 5 articles, the engine starts looking at 
    similar users to pull articles seen by those, 
    among which only those predicted to be liked by the user via SVD factorization 
    of the user-item matrix are shown.
    
    '''
    def __init__(self):
        ''' Instantiates a recommender object, load the datasets and 
        pre-process them to have the recommender function ready
        '''
        # load datasets
        self.df = pd.read_csv('data/user-item-interactions.csv')
        self.df_content = pd.read_csv('data/articles_community.csv')
        del self.df['Unnamed: 0']
        del self.df_content['Unnamed: 0']
        
        # convert all user and article ids to same format
        self.df['article_id'] = self.df['article_id'].astype('float')
        self.df['article_id'] = self.df['article_id'].astype('int')
        self.df['article_id'] = self.df['article_id'].astype('str')
        self.df_content['article_id'] = self.df_content['article_id'].astype('str')
         
        # Remove any rows that have the same article_id - only keep the first
        self.df_content.drop_duplicates(subset='article_id', 
                                        keep='first', 
                                        inplace=True)
        
        # map user email to create new user_id column
        email_encoded = f.email_mapper(self.df)
        del self.df['email']
        self.df['user_id'] = email_encoded
        
        
        # create user_item matrix
        self.user_item_matrix = f.create_user_item_df(self.df) 
        
        # create users similarity (dot_product) dataframe
        users_dot_prod_np = np.dot(self.user_item_matrix,
                                   np.transpose(self.user_item_matrix))
        self.dot_prod_df = pd.DataFrame(users_dot_prod_np, 
                                        columns = self.user_item_matrix.index, 
                                        index = self.user_item_matrix.index)
        
        # get users that have interacted with more than 5 articles 
        self.old_users = self.user_item_matrix[self.user_item_matrix.sum(axis=1)>5].index.tolist()
        
        # get users that have not seen any content:
        self.new_users = self.user_item_matrix[self.user_item_matrix.sum(axis=1)==0].index.tolist()    

        # get users that have interacted with 5 articles or less
        self.recent_users = list(set(self.user_item_matrix[self.user_item_matrix.sum(axis=1)<6].index.tolist()) - set(self.new_users))     
        
    def make_recommendations(self, user_id, m=10, k=50):
        ''' Makes recommendations for a given user_id
        
        INPUT:
            user_id (int): user for whom to make recommendations
            m (int): number of recommendations to show
            k (int): number of latent factors for SVD factorization
        OUTPUT
            None, but stores the titles of recommended articles in the 
            'recommended_articles' attribute
        '''
        self.recommended_articles = f.final_recommender(user_id, self.df, 
                                                        self.df_content, 
                                                        self.user_item_matrix, 
                                                        self.dot_prod_df, 
                                                        m=m, 
                                                        k=k)

    
    def get_articles_seen(self, user_id):
        ''' Returns titles of articles seen by user
        
        INPUT:
            user_id (int): user for whom to retrieve articles seen
        
        OUTPUT:
            None, stores articles seen in 'articles_seen' attribute
        '''
        articles_ids = f.get_user_articles(user_id, 
                                           self.user_item_matrix)
        articles_names = f.get_article_names(articles_ids, 
                                             self.df, 
                                             self.df_content) 
        self.articles_seen = [articles_ids,articles_names]

    
    def get_article_data(self, article_id):
        ''' Return title, teaser and body text of an article id
        
        INPUT:
            article_id (int): id of article for which to return data
        
        OUTPUT:
            None, but stores article title, teaser and body text as attributes
        '''
        try:
            self.article_title = self.df_content[self.df_content['article_id']==article_id]['doc_full_name'].iloc[0]
        except:
            self.article_title = self.df[self.df['article_id']==article_id]['title'].iloc[0].title()
        try:
            self.article_teaser = self.df_content[self.df_content['article_id']==article_id]['doc_description'].iloc[0]
        except:
            self.article_teaser = None
        try:
            self.article_body = self.df_content[self.df_content['article_id']==article_id]['doc_body'].iloc[0]
        except:
            self.article_body = None
    
    
    def record_user_activity(self,user_id,article_id):    
        ''' Stores interaction of given user with given article and rebuilds
        data structures to compute user similarity and collaborative filtering
        based recommendations
        
        INPUT:
            user_id (int): user to store activity
            article_id (int): article read by user
            
        OUTPUT:
            None. Adds a new row at then end of df with the new activity data
            and updates user_item_matrix and user_similarity matrix
        '''
        # add new row to df
        try:
            new_row = [article_id, self.df[self.df['article_id']==article_id]['title'].iloc[0], user_id]
        except:
            new_row = [article_id, self.df_content[self.df_content['article_id']==article_id]['doc_full_name'].iloc[0], user_id]
        self.df.loc[self.df.shape[0]] = new_row
        # update user_item_matrix and user_similarity matrix
        self.user_item_matrix = f.create_user_item_df(self.df) 
        users_dot_prod_np = np.dot(self.user_item_matrix,np.transpose(self.user_item_matrix))
        self.dot_prod_df = pd.DataFrame(users_dot_prod_np, 
                                        columns = self.user_item_matrix.index, 
                                        index = self.user_item_matrix.index)
        
        