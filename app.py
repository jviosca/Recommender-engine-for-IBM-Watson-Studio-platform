import streamlit as st
import numpy as np
from recommender import Recommender
import base64
from PIL import Image

#################################################
#  Initialize objects, variables and functions  #
#################################################

@st.cache_data
def start_engine():
    ''' Initializes Recommender class object
    Parameters
    -------
    None

    Returns
    -------
    Recommender (python class): object that stores user activity, content data
    and makes recommendations using several data structures as attributes 
    and methods to compute them.

    '''
    return Recommender()    

# Instantiate recommender object
rec_engine = start_engine()

# store rec_engine object in session state
# needed by streamlit to be able to update the 'df' attribute of Recommender class
if "rec_engine" not in st.session_state:
    st.session_state.rec_engine = rec_engine

def update_user_type():
    ''' Updates user type if new articles are added to the seen list
    New user = has seen 0 articles
    Recent user = has seen 5 articles or fewer
    Old user = has seen more than 5 articles
    
    Parameters:
    -------
    None

    Returns
    -------
    None.

    '''
    # Compute articles seen by user
    st.session_state.rec_engine.get_articles_seen(st.session_state.user_id)
    articles_seen = st.session_state.rec_engine.articles_seen[0]
    if len(articles_seen) > 0:
        st.session_state.user_type = "Recent user"
    if len(articles_seen) > 5:
        st.session_state.user_type = "Old user"    

def set_user_id():
    ''' Sets user id to a constant value througout session (while browser 
    is not refreshed), to avoid changing user_id value upon interaction
    with input elements of the app (slider or the tabs)
    It is called by 'user_type' selectbox callback
    
    Parameters
    -------
    None

    Returns
    -------
    None.
    '''
    if st.session_state.user_type == 'Recent user':
       st.session_state.user_id = np.random.choice(st.session_state.rec_engine.recent_users)
    elif st.session_state.user_type == 'Old user':
       st.session_state.user_id = np.random.choice(st.session_state.rec_engine.old_users)
    else:
        new_id = str(int(st.session_state.rec_engine.df['user_id'].unique()[-1])+1)  
        st.session_state.user_id = new_id  
    update_user_type()
   
# Set user_id session state. Default to 'New user'
if 'user_id' not in st.session_state:
   st.session_state.user_type = 'New user' 
   set_user_id()
    
def store_new_seen(article_id):
    ''' Stores user activity and updates data structures to recompute 
    similar articles and users
    
    Parameters:
    -------
    article_id (int): id of article recently seen

    Returns
    -------
    None.

    '''
    st.session_state.rec_engine.record_user_activity(st.session_state.user_id, 
                                                     article_id)
    # if number of articles seen > 5, user type needs to change:
    update_user_type()

@st.cache_data
def show_gif(path, alt_text):
    ''' Shows GIF in Streamlit page

    Parameters
    ----------
    path (string): path to gif file
    alt_text (string): alt text for gif

    Returns
    -------
    None.

    '''
    
    gif_file_ = open(path, "rb")
    gif_contents = gif_file_.read()
    data_url = base64.b64encode(gif_contents).decode("utf-8")
    gif_file_.close()
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="' 
                + alt_text + '" style="max-width:100%">',
                    unsafe_allow_html=True)   

@st.cache_data
def show_image(path):
    ''' Places image centered
    

    Parameters
    ----------
    path (string): path to image file

    Returns
    -------
    None.

    '''
    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        image = Image.open(path)
        st.image(image)
    
    
###############
#   Layout    #
###############

# Page title
st.title(':blue[Recommender system for IBM Watson Studio platform]')

# There are 3 tabs: Home, Recommender and Credits
tab1, tab2, tab3, tab4 = st.tabs(['Home', 'How it works', 'Recommender', 'Credits'])
with tab1:
    show_image('img/like.png') 
    st.write('This webapp showcases a **recommender system** built from records \
            of user interactions with content items at \
            [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio).')      
    st.write('The engine recommends content items using **content-based and \
             collaborative-filtering** approaches tailored to different user types:')
    st.markdown('- **New users**: users that have not interacted with any \
                content in the platform yet. A list of the top 10 most \
                *popular articles* are recommended in this situation.')
    st.markdown('- **Recent users**: users that have interacted with 5 items \
                or less. In addition to popular articles, articles that are \
                similar to those previously seen by the user are also offered \
                in the recommendations list. *Similar articles* are pulled \
                using **Natural Language Processing** to identify those \
                articles that share more words in their title text.')
    st.markdown('- **Old users**: those that have seen more than 5 items. In \
                addition to popular and similar articles (content based \
              recommendations) just explained, \
            articles read by other *similar users* are also pulled as \
                recommendations.')
    show_image('img/update.png')
    st.write('The list of recommendations is **updated dynamically** according \
             to user activity. To interact with the engine and see its results, \
             click on the *Recommender* tab above and simulate user activity on the \
             platform by picking articles to read. You can obtain \
             personalized recommendations for you or other users!')
    show_gif('img/app_tabs.gif', 'tabs')

with tab2:
    show_image('img/use.png') 
    st.write('To obtain recommendations, follow these steps:')
    st.markdown('1. **Choose a user type** (to set a random "user id" from that \
                category). Articles that have been seen by that user (if any) \
                appear at the top. Recommendations appear below. Each item \
                is contained in a expander - unfold it to read the article teaser \
                by clicking on the arrow located next to the title.')
    with st.expander('Show GIF'):
        show_gif('img/select_user.gif', 'user_type gif')
    show_image('img/see.png')
    st.markdown('2. To obtain new recommendations, add items to the \
                *Articles seen* list by clicking on the **Mark as seen** button \
                below the teaser from the *Recommended articles* section \
                (you need to expand the item first).')
    with st.expander('Show GIF'):
        show_gif('img/mark_as_seen.gif', 'new_seen_item gif')
    st.write("")
    st.write('For users with a lot of activity, \
                 a **slider** lets you control the number of read items to show\
                 (only visible when the user has seen more than 6 articles).')
    with st.expander('Show GIF'):
        show_gif('img/slider.gif', 'slider gif')    

with tab3:
    show_image('img/play.png')
    st.session_state.rec_engine.get_articles_seen(st.session_state.user_id)
    articles_seen = st.session_state.rec_engine.articles_seen
    number_seen = len(articles_seen[0])
    
    # Choose user type. Change user_id only if selectbox changes:
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.selectbox('User type:', 
                     ('New user', 'Recent user', 'Old user'), 
                     key = 'user_type', 
                     on_change=set_user_id)
    with col2:
        st.text_input('User ID:',
                      value = st.session_state.user_id,
                      disabled = True)
    with col3:
        st.text_input('Num. articles seen:',
                      value = number_seen,
                      disabled = True)
    with col4:
        # calculate n_users
        if st.session_state.user_type == 'Old user':
            n_users = len(st.session_state.rec_engine.old_users)
        elif st.session_state.user_type == 'Recent user':
            n_users = len(st.session_state.rec_engine.recent_users)
        else:
            n_users = len(st.session_state.rec_engine.new_users)
        st.text_input('Num. of ' + (st.session_state.user_type) + 's:',
                      value = n_users,
                      disabled = True)        
    
    # Section 'Articles seen by user'
    if number_seen>0:
        st.markdown('##### :blue[Articles seen by selected user:]')
    
        # set slider step 
        if number_seen % 2 != 0: # uneven number
            slider_step = 1
        else: # even number
            slider_step = 2
        if number_seen % 6 == 0: # multiples of 6
            slider_step = 6
        if number_seen>6:
            # Slider to show more articles
            num_articles_shown = st.slider('Number of seen articles to show:',
                                           min_value = 6,
                                           max_value = number_seen,
                                           step = slider_step,
                                           key = 'slider_val')
        else: # New users and recent users            
            num_articles_shown = number_seen
        
        # Layout articles
        col1,col2 = st.columns(2)
        for idx,article_id in enumerate(articles_seen[0][:num_articles_shown]):
            st.session_state.rec_engine.get_article_data(article_id)
            if (idx+1) % 2 == 0: # even idx (2,4,6, etc) in column 2
                with col2:
                    with st.expander('**' + 
                                     st.session_state.rec_engine.article_title +
                                     '**'):
                        st.write(st.session_state.rec_engine.article_teaser)
            else:
                with col1:
                    with st.expander('**' + 
                                     st.session_state.rec_engine.article_title + 
                                     '**'):
                        st.write(st.session_state.rec_engine.article_teaser)
        
    
    # Section 'Recommendations'
    st.markdown('##### :blue[Recommended articles for selected user:]')
    st.session_state.rec_engine.make_recommendations(st.session_state.user_id)
    recommendations = st.session_state.rec_engine.recommended_articles[0]
    
    # Layout recommendations
    col1,col2 = st.columns(2)
    for idx,article_id in enumerate(recommendations):
        st.session_state.rec_engine.get_article_data(article_id)
        if (idx+1) % 2 == 0: # even idx (2,4,6, etc) in column 2
            with col2:
                with st.expander('**' + 
                                 st.session_state.rec_engine.article_title + 
                                 '**'):
                    st.write(st.session_state.rec_engine.article_teaser)
                    st.button('Mark as seen', 
                              article_id, 
                              on_click=store_new_seen, 
                              args=(article_id,))
        else:
            with col1:
                with st.expander('**' + 
                                 st.session_state.rec_engine.article_title + 
                                 '**'):
                    st.write(st.session_state.rec_engine.article_teaser)
                    st.button('Mark as seen', 
                              article_id, 
                              on_click=store_new_seen, 
                              args=(article_id,))

with tab4:
    st.write('For more information, visit the [GitHub repository](https://github.com/jviosca/Recommender-engine-for-IBM-Watson-Studio-platform).')  
    st.write('Author: [Jose Viosca Ros](https://jvros.com.es/index.php/es/inicio/)')           