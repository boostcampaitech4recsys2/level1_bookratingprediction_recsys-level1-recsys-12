import os
import numpy as np
import pandas as pd
import re

def process_user_data():
    """Preprocess users.csv and save it at './data/processed/users.csv'"""
    # load data
    load_path = './data/'
    users = pd.read_csv(load_path + 'users.csv')
    
    # age 는 sampling with replacement로 impute
    users.loc[users['age'].isna(), 'age'] = np.random.choice(users.loc[users['age'].notna(), 'age'], 
                                                             users['age'].isna().sum())

    # str split users['location'] to city, state and country
    # codes from mission 1
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') 

    users['location_city']    = users['location'].apply(lambda x: x.split(',')[0].strip())
    users['location_state']   = users['location'].apply(lambda x: x.split(',')[1].strip())
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

    users = users.replace('na', np.nan) 
    users = users.replace('',   np.nan)
    
    # country가 na지만 states 정보가 있는 경우 이 정보를 활용해서 country 채워넣기
    states_with_null = users[(users['location_state'].notnull()) & 
                             (users['location_country'].isna())]['location_state'].values

    for state in states_with_null:
        try:
            country = users.loc[(users['location'].str.contains(state)), 'location_country'].value_counts().index[0]
            users.loc[(users['location'].str.contains(state)) & 
                      (users['location_country'].isna()), 'location_country'] = country
        except:
            pass

    # country가 na지만 city 정보가 있는 경우 이 정보를 활용해서 country 채워넣기
    cities_with_null = users[(users['location_city'].notnull())  & 
                             (users['location_country'].isna())]['location_city'].values

    for city in cities_with_null:
        try:
            country = users.loc[(users['location'].str.contains(city)), 'location_country'].value_counts().index[0]
            users.loc[(users['location'].str.contains(city)) & 
                      (users['location_country'].isna()), 'location_country'] = country
        except:
            pass

    # england, california 같은 정보가 country에 있는 경우가 있는데
    # 각 나라 별로 users['location']의 최빈값으로 대체
    countries_list = users['location_country'].value_counts()
    for country in countries_list.index:
        try:
            new_country = users.loc[(users['location'].str.contains(country)), 'location_country'].value_counts().index[0]
            users.loc[(users['location'].str.contains(country)) & (users['location_country'] == country), 
                      'location_country'] = new_country
        except:
            pass
    
    # country가 threshold보다 작은 경우를 다합쳐서 others로 대체
    threshold = 30

    others_list = users['location_country'].value_counts()[users['location_country'].value_counts() < threshold].index
    for country in others_list:
        try:
            users.loc[(users['location_country'] == country), 'location_country'] = 'others'
        except:
            pass
    
    # 그래도 country가 NaN인 경우 최빈값인 random sampling으로 impute
    random_country = np.random.choice(users.loc[users['location_country'].notna(), 'location_country'], 
                                      users['location_country'].isna().sum())
    users.loc[users['location_country'].isna(), 'location_country'] = random_country
    
    # inpute된 country 활용해서 state, city 채워넣기
    country_list = users['location_country'].value_counts().index

    for country in country_list:
        try:
            random_state = np.random.choice(
                users.loc[(users['location_country'] == country) & (users['location_state'].notna()), 'location_state'],
                users.loc[(users['location_country'] == country), 'location_state'].isna().sum()
            )
            users.loc[(users['location_country'] == country) & (users['location_state'].isna()), 'location_state'] = random_state
            
            state_list = users.loc[(users['location_country'] == country), 'location_state'].value_counts().index
            # state_list = state_list[state_list > 1].index
            for state in state_list:
                random_city = np.random.choice(
                    users.loc[(users['location_country'] == country) & 
                              (users['location_state']   == state)   & 
                              (users['location_city'].notna()), 'location_city'],
                    users.loc[(users['location_country'] == country) & (users['location_state'] == state), 'location_city'].isna().sum()
                )
                users.loc[(users['location_country'] == country) & 
                          (users['location_state'] == state)     & 
                          (users['location_city'].isna()), 'location_city'] = random_city
        except:
            pass

    for country in country_list:
        try:
            random_city = np.random.choice(
                users.loc[(users['location_country'] == country) & (users['location_city'].notna()), 'location_city'],
                users.loc[(users['location_country'] == country), 'location_city'].isna().sum()
            )
            users.loc[(users['location_country'] == country) & (users['location_city'].isna()), 'location_city'] = random_city
        except:
            pass

    # drop columns and save processed users.csv
    users.drop(['location'], axis=1, inplace=True)
    
    data_path = './data/processed/'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    users.to_csv(data_path + 'users.csv', index=False)
    
        
def process_book_data():
    """Preprocess books.csv and save it at './data/processed/books.csv'"""
    load_path = './data/original/'
    books = pd.read_csv(load_path + 'books.csv')
    
    # impute language based on first char of isbn
    null_lang = books.loc[books['language'].isna(), 'isbn'].apply(lambda x: x[:1]).value_counts().to_dict()

    for i in range(10):
        i = str(i)
        possible_lang = books.loc[
            (books['isbn'].apply(lambda x: x[:1]) == i) & (books['language'].notna()), 
            'language'].values
        try:
            books.loc[(books['isbn'].apply(lambda x: x[:1]) == i) & (books['language'].isna()),
                    'language'] = np.random.choice(possible_lang, null_lang[i])
        except:
            pass
    
    # 남은 na lang은 랜덤하게 추출
    random_lang = np.random.choice(books['language'], books['language'].isna().sum())
    books.loc[(books['language']).isna(), 'language'] = random_lang
    
    # publisher 전처리
    books['publisher'] = books['publisher'].str.replace("'s", 's') # reader's digest 같은 출판사 이름을 간단하게
    books['publisher'] = books['publisher'].str.replace("s'", 's')

    books.loc[books[books['publisher'].notnull()].index, 'publisher'] = books[books['publisher'].notnull()]['publisher'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['publisher'] = books['publisher'].str.lower()
    books['publisher'] = books['publisher'].str.strip()

    # 출판한 책이 10개 이상되는 출판사로 출판사 이름을 포함하는 경우 이름 바꿔주기
    # 예 penguin books ltd 를 penguin books 를 포함하니까 penguin books로 바뀜
    threshold = 10

    publisher_list = books['publisher'].value_counts()[books['publisher'].value_counts() > threshold].index
    for publisher in publisher_list:
        try:
            books.loc[books['publisher'].str.contains(publisher), 'publisher'] = publisher
        except:
            pass
    
    # 책을 threshold보다 적게 출판한 출판사는 others로 바꾸기
    threshold = 10

    publisher_list = books['publisher'].value_counts()[books['publisher'].value_counts() > threshold].index

    books.loc[books['publisher'].notna() & books['publisher'].apply(lambda x: x not in publisher_list), 'publisher'] = 'others'
    books['publisher'].value_counts()

    # category 전처리
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()
    books['category'].value_counts()

    # 대표적인 카테고리와 상위 카테고리를 만들기
    # 다른걸 추가하거나 분류를 바꿔보는 것도 가능
    categories = {   
        'animal'         : ['animal', 'bird', 'pets', 'cats', 'dogs', 'bears', 'dino'],
        'arts'           : ['art', 'photography', 'architecture', 'music', 'criticism', 'perform', 'design', 'paint', 
                            'decorat', 'draw', 'act', 'picture', 'author', 'composer'],
        'biographies'    : ['biography', 'memoir'],
        'business'       : ['business', 'money', 'economic', 'finance', 'invest', 'management', 'sales', 'marketing'],
        'comic'          : ['comic', 'graphic'],
        'computer'       : ['computer', 'technology', 'software'],
        'cook'           : ['cook', 'food', 'wine', 'baking', 'desserts', 'beverage', 'alcohol'],
        'education'      : ['education', 'teach', 'test', 'study', 'book'],
        'engineering'    : ['engineer', 'transportation', 'electronic'],
        'entertainment'  : ['humor', 'entertainment', 'game'],
        'family'         : ['child', 'famil', 'parent', 'relationship', 'marriage', 'baby', 'wedding', 'brother', 
                            'sister', 'boy', 'girl', 'aunt'],
        'health'         : ['health', 'fitness', 'diet', 'body', 'mind'],
        'history'        : ['history'],
        'hobby'          : ['craft', 'hobby', 'home', 'garden', 'landscape', 'collect'],
        'juvenile'       : ['student', 'school', 'teen', 'young', 'juvenile', 'friendship', 'adolescence'],
        'law'            : ['law', 'legal', 'divorce'],
        'life'           : ['life'],
        'medical'        : ['medical', 'pharmacology', 'medicine', 'dentistry', 'disease'],
        'mystery'        : ['mystery', 'extraterrestrial', 'fairy', 'curiosit', 'wonder', 'magic', 'ghost'],
        'reference'      : ['reference'],
        'religion'       : ['christian', 'bible', 'religion', 'spirit', 'church', 'catholic', 'angel', 'buddhism', 
                            'bereavement'],
        'self_help'      : ['help', 'interpersonal', 'relation', 'behavior', 'love'],
        'sports'         : ['sport', 'outdoor'],
        'thriller'       : ['thriller', 'suspense', 'crim', 'horror', 'murder', 'death'],
        'travel'         : ['travel', 'voyage'],
        'world'          : ['english', 'england', 'australia', 'brit', 'africa', 'states', 'france', 'canada', 'america', 
                            'china', 'egypt', 'germa', 'ireland', 'california', 'europe'],
        'social_science' : ['social', 'politic', 'psychology', 'philosophy', 'politic', 'government', 'geography',],
        'science'        : ['science', 'nature', 'math'],
        'literature'     : ['literature', 'science fiction', 'fiction', 'fantasy', 'drama', 'poetry', 'stories', 
                            'collections', ' fairy tale', 'horror', 'romance', 'adultery', 'adventure'],
    }

    # 기존 category에 해당 단어를 포함하면 해당 단어로 덮어쓰기
    # 예 auto-biography -> biography
    books['category_high'] = np.NaN

    for category_high in categories.keys():
        for category in categories[category_high]:
            books.loc[books['category'].notna() & books['category'].str.contains(category), 'category_high'] = category_high
            books.loc[books['category'].notna() & books['category'].str.contains(category), 'category']      = category

    # author와 title 전처리

    books['book_author'] = books['book_author'].apply(lambda x: re.sub('[\W_]+', ' ', x).strip())
    books['book_author'] = books['book_author'].str.lower()
    books['book_author'] = books['book_author'].str.strip()

    books['book_title']  = books['book_title'].apply(lambda x: re.sub('[\W_]+', ' ', x).strip())
    books['book_title']  = books['book_title'].str.lower()
    books['book_title']  = books['book_title'].str.strip()

    # author가 같은 책이면 category도 비슷할 것이라고 믿고 impute하기
    no_cat_authors = books.loc[books['category_high'].isna(), 'book_author'].value_counts()
    no_cat_authors = no_cat_authors[no_cat_authors > 1].to_dict()

    for author in no_cat_authors:
        try:
            cat_list      = books.loc[(books['category_high'].notna()) & (books['book_author'] == author),
                                    'category'].values
            cat_high_list = books.loc[(books['category_high'].notna()) & (books['book_author'] == author),
                                    'category_high'].values
            cats      = np.random.choice(cat_list,      no_cat_authors[author])
            cat_highs = np.random.choice(cat_high_list, no_cat_authors[author])
            
            books.loc[(books['category_high'].isna()) & (books['book_author'] == author), 'category']      = cats
            books.loc[(books['category_high'].isna()) & (books['book_author'] == author), 'category_high'] = cat_highs
        except:
            pass

    # 마지막으로 publisher가 같은 책이면 category도 비슷할 것이라고 믿고 impute하기
    no_cat_publishers = books.loc[books['category_high'].isna(), 'publisher'].value_counts()
    no_cat_publishers = no_cat_publishers[no_cat_publishers > 1].to_dict()

    for publisher in no_cat_publishers:
        try:
            cat_list      = books.loc[(books['category_high'].notna()) & (books['publisher'] == publisher),
                                    'category'].values
            cat_high_list = books.loc[(books['category_high'].notna()) & (books['publisher'] == publisher),
                                    'category_high'].values
            cats      = np.random.choice(cat_list,      no_cat_publishers[publisher])
            cat_highs = np.random.choice(cat_high_list, no_cat_publishers[publisher])
            
            books.loc[(books['category_high'].isna()) & (books['publisher'] == publisher), 'category']      = cats
            books.loc[(books['category_high'].isna()) & (books['publisher'] == publisher), 'category_high'] = cat_highs
        except:
            pass
    
    books.loc[books['category_high'].isna(), 'category'] = np.NaN
    books.loc[books['category_high'].isna(), 'category_high'] = np.random.choice(books.loc[books['category_high'].notna(), 'category_high'],
                                                                                books['category_high'].isna().sum())

    for category_high in categories.keys():
        try:
            cat_list = np.random.choice(categories[category_high], 
                                        books.loc[(books['category_high'] == category_high), 'category'].isna().sum())
            books.loc[(books['category_high'] == category_high) & 
                    (books['category'].isna()), 'category'] = cat_list
        except:
            pass

    # save processed users.csv
    data_path = './data/processed/'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    books.to_csv(data_path + 'books.csv', index=False)

if __name__ == "__main__":
    print(f'--------------- Processing users.csv ---------------')
    process_user_data()
    print(f'--------------- Processing books.csv ---------------')
    process_book_data() 
    print(f'---------------         Done         ---------------')