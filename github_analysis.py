import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load the Data
# ----------------------------

# Read users.csv and repositories.csv
users_df = pd.read_csv('users.csv', dtype=str)
repos_df = pd.read_csv('repositories.csv', dtype=str)

# Convert relevant columns to appropriate data types
# Added 'utc=True' to make datetime objects timezone-aware
users_df['followers'] = pd.to_numeric(users_df['followers'], errors='coerce').fillna(0).astype(int)
users_df['following'] = pd.to_numeric(users_df['following'], errors='coerce').fillna(0).astype(int)
users_df['public_repos'] = pd.to_numeric(users_df['public_repos'], errors='coerce').fillna(0).astype(int)
users_df['created_at'] = pd.to_datetime(users_df['created_at'], errors='coerce', utc=True)

repos_df['stargazers_count'] = pd.to_numeric(repos_df['stargazers_count'], errors='coerce').fillna(0).astype(int)
repos_df['watchers_count'] = pd.to_numeric(repos_df['watchers_count'], errors='coerce').fillna(0).astype(int)
repos_df['created_at'] = pd.to_datetime(repos_df['created_at'], errors='coerce', utc=True)

# ----------------------------
# Question 1
# ----------------------------
# 1. Who are the top 5 users in Zurich with the highest number of followers? 
#    List their login in order, comma-separated.

def question_1(users_df):
    top_5 = users_df.sort_values(by='followers', ascending=False).head(5)
    logins = top_5['login'].tolist()
    result = ','.join(logins)
    print(f"1. Top 5 users by followers: {result}")

# ----------------------------
# Question 2
# ----------------------------
# 2. Who are the 5 earliest registered GitHub users in Zurich? 
#    List their login in ascending order of created_at, comma-separated.

def question_2(users_df):
    earliest_5 = users_df.sort_values(by='created_at').head(5)
    logins = earliest_5['login'].tolist()
    result = ','.join(logins)
    print(f"2. 5 earliest registered users: {result}")

# ----------------------------
# Question 3
# ----------------------------
# 3. What are the 3 most popular licenses among these users? 
#    Ignore missing licenses. List the license_name in order, comma-separated.

def question_3(repos_df):
    licenses = repos_df['license_name'].dropna()
    licenses = licenses[licenses != '']  # Ignore empty strings
    top_3 = licenses.value_counts().head(3).index.tolist()
    result = ','.join(top_3)
    print(f"3. Top 3 licenses: {result}")

# ----------------------------
# Question 4
# ----------------------------
# 4. Which company do the majority of these developers work at?
#    Company (cleaned up as explained above)

def question_4(users_df):
    companies = users_df['company'].dropna()
    companies = companies[companies != '']  # Exclude empty strings
    if companies.empty:
        result = "No company data available"
    else:
        top_company = companies.value_counts().idxmax()
        result = top_company
    print(f"4. Majority company: {result}")

# ----------------------------
# Question 5
# ----------------------------
# 5. Which programming language is most popular among these users?
#    Language

def question_5(repos_df):
    languages = repos_df['language'].dropna()
    languages = languages[languages != '']  # Exclude empty strings
    if languages.empty:
        result = "No language data available"
    else:
        top_language = languages.value_counts().idxmax()
        result = top_language
    print(f"5. Most popular language: {result}")

# ----------------------------
# Question 6
# ----------------------------
# 6. Which programming language is the second most popular among users who joined after 2020?
#    Language

def question_6(users_df, repos_df):
    # Filter users who joined after 2020
    comparison_timestamp = pd.Timestamp('2020-12-31', tz='UTC')  # Make timezone-aware
    filtered_users = users_df[users_df['created_at'] > comparison_timestamp]
    filtered_logins = filtered_users['login']
    
    # Filter repositories of these users
    filtered_repos = repos_df[repos_df['login'].isin(filtered_logins)]
    
    # Count languages
    languages = filtered_repos['language'].dropna()
    languages = languages[languages != '']  # Exclude empty strings
    if languages.empty or languages.nunique() < 2:
        result = "Not enough language data"
    else:
        top_languages = languages.value_counts().index.tolist()
        if len(top_languages) < 2:
            result = top_languages[0]
        else:
            result = top_languages[1]
    print(f"6. Second most popular language among users who joined after 2020: {result}")

# ----------------------------
# Question 7
# ----------------------------
# 7. Which language has the highest average number of stars per repository?
#    Language

def question_7(repos_df):
    # Exclude repositories with missing language
    repos = repos_df[repos_df['language'].notna() & (repos_df['language'] != '')]
    if repos.empty:
        result = "No language data available"
    else:
        avg_stars = repos.groupby('language')['stargazers_count'].mean()
        top_language = avg_stars.idxmax()
        result = top_language
    print(f"7. Language with highest average stars: {result}")

# ----------------------------
# Question 8
# ----------------------------
# 8. Let's define leader_strength as followers / (1 + following). 
#    Who are the top 5 in terms of leader_strength? List their login in order, comma-separated.

def question_8(users_df):
    # Avoid division by zero by adding 1 to following
    users_df['leader_strength'] = users_df.apply(
        lambda row: row['followers'] / (1 + row['following']) if row['following'] >= 0 else 0, axis=1
    )
    top_5 = users_df.sort_values(by='leader_strength', ascending=False).head(5)
    logins = top_5['login'].tolist()
    result = ','.join(logins)
    print(f"8. Top 5 users by leader_strength: {result}")

# ----------------------------
# Question 9
# ----------------------------
# 9. What is the correlation between the number of followers and the number of public repositories among users in Zurich?
#    Correlation between followers and repos (to 3 decimal places, e.g. 0.123 or -0.123)

def question_9(users_df):
    followers = users_df['followers']
    public_repos = users_df['public_repos']
    if followers.empty or public_repos.empty:
        corr = 0
    else:
        corr, _ = pearsonr(followers, public_repos)
    corr_rounded = round(corr, 3)
    print(f"9. Correlation between followers and public repos: {corr_rounded}")

# ----------------------------
# Question 10
# ----------------------------
# 10. Does creating more repos help users get more followers? 
#     Using regression, estimate how many additional followers a user gets per additional public repository.
#     Regression slope of followers on repos (to 3 decimal places, e.g. 0.123 or -0.123)

def question_10(users_df):
    X = users_df[['public_repos']].values
    y = users_df['followers'].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    slope_rounded = round(slope, 3)
    print(f"10. Regression slope of followers on repos: {slope_rounded}")

# ----------------------------
# Question 11
# ----------------------------
# 11. Do people typically enable projects and wikis together? 
#     What is the correlation between a repo having projects enabled and having wiki enabled?
#     Correlation between projects and wiki enabled (to 3 decimal places, e.g. 0.123 or -0.123)

def question_11(repos_df):
    # Convert 'has_projects' and 'has_wiki' to numeric (1 for true, 0 for false)
    repos_df['has_projects_num'] = repos_df['has_projects'].map({'true':1, 'false':0})
    repos_df['has_wiki_num'] = repos_df['has_wiki'].map({'true':1, 'false':0})
    
    # Drop rows with missing values
    df = repos_df[['has_projects_num', 'has_wiki_num']].dropna()
    
    if df.empty:
        corr = 0
    else:
        corr, _ = pearsonr(df['has_projects_num'], df['has_wiki_num'])
    corr_rounded = round(corr, 3)
    print(f"11. Correlation between projects and wiki enabled: {corr_rounded}")

# ----------------------------
# Question 12
# ----------------------------
# 12. Do hireable users follow more people than those who are not hireable?
#     Average of following per user for hireable=true minus the average following for the rest (to 3 decimal places, e.g. 12.345 or -12.345)

def question_12(users_df):
    # Convert 'hireable' to boolean
    users_df['hireable_bool'] = users_df['hireable'].map({'true': True, 'false': False, '': False})
    
    # Calculate averages
    hireable_avg = users_df[users_df['hireable_bool'] == True]['following'].mean()
    non_hireable_avg = users_df[users_df['hireable_bool'] == False]['following'].mean()
    
    difference = hireable_avg - non_hireable_avg
    difference_rounded = round(difference, 3)
    print(f"12. Average following difference (hireable - non-hireable): {difference_rounded}")

# ----------------------------
# Question 13
# ----------------------------
# 13. Some developers write long bios. Does that help them get more followers? 
#     What's the impact of the length of their bio (in Unicode words, split by whitespace) with followers? 
#     (Ignore people without bios)
#     Regression slope of followers on bio word count (to 3 decimal places, e.g. 12.345 or -12.345)

def question_13(users_df):
    # Filter users with non-empty bios
    users_with_bio = users_df[users_df['bio'].notna() & (users_df['bio'] != '')].copy()
    
    if users_with_bio.empty:
        slope = 0
    else:
        # Calculate word count
        users_with_bio['bio_word_count'] = users_with_bio['bio'].apply(lambda x: len(x.split()))
        
        X = users_with_bio[['bio_word_count']].values
        y = users_with_bio['followers'].values
        
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
    
    slope_rounded = round(slope, 3)
    print(f"13. Regression slope of followers on bio word count: {slope_rounded}")

# ----------------------------
# Question 14
# ----------------------------
# 14. Who created the most repositories on weekends (UTC)? 
#     List the top 5 users' login in order, comma-separated

def question_14(repos_df):
    # Determine if the repository was created on a weekend
    repos_df['created_weekday'] = repos_df['created_at'].dt.weekday  # Monday=0, Sunday=6
    repos_df['is_weekend'] = repos_df['created_weekday'].apply(lambda x: 1 if x >=5 else 0)
    
    # Filter repositories created on weekends
    weekend_repos = repos_df[repos_df['is_weekend'] == 1]
    
    # Count number of weekend repos per user
    weekend_counts = weekend_repos['login'].value_counts().head(5).index.tolist()
    result = ','.join(weekend_counts)
    print(f"14. Top 5 users by weekend repo creations: {result}")

# ----------------------------
# Question 15
# ----------------------------
# 15. Do people who are hireable share their email addresses more often?
#     [fraction of users with email when hireable=true] minus [fraction of users with email for the rest] 
#     (to 3 decimal places, e.g. 0.123 or -0.123)

def question_15(users_df):
    # Convert 'hireable' to boolean
    users_df['hireable_bool'] = users_df['hireable'].map({'true': True, 'false': False, '': False})
    
    # Calculate fractions
    hireable = users_df[users_df['hireable_bool'] == True]
    non_hireable = users_df[users_df['hireable_bool'] == False]
    
    hireable_fraction = hireable['email'].notna() & (hireable['email'] != '')
    hireable_fraction = hireable_fraction.mean()
    
    non_hireable_fraction = non_hireable['email'].notna() & (non_hireable['email'] != '')
    non_hireable_fraction = non_hireable_fraction.mean()
    
    difference = hireable_fraction - non_hireable_fraction
    difference_rounded = round(difference, 3)
    print(f"15. Fraction difference in email sharing (hireable - non-hireable): {difference_rounded}")

# ----------------------------
# Question 16
# ----------------------------
# 16. Let's assume that the last word in a user's name is their surname 
#     (ignore missing names, trim and split by whitespace.)
#     What's the most common surname? 
#     (If there's a tie, list them all, comma-separated, alphabetically)

def question_16(users_df):
    # Filter users with non-empty names
    users_with_name = users_df[users_df['name'].notna() & (users_df['name'] != '')].copy()
    
    if users_with_name.empty:
        result = "No name data available"
    else:
        # Extract surnames
        users_with_name['surname'] = users_with_name['name'].apply(lambda x: x.strip().split()[-1] if len(x.strip().split()) > 0 else '')
        surnames = users_with_name['surname']
        surname_counts = surnames.value_counts()
        if surname_counts.empty:
            result = "No surname data available"
        else:
            max_count = surname_counts.max()
            most_common = surname_counts[surname_counts == max_count].index.tolist()
            most_common_sorted = sorted(most_common)
            result = ','.join(most_common_sorted)
    print(f"16. Most common surname(s): {result}")

# ----------------------------
# Execute All Questions
# ----------------------------

def main():
    question_1(users_df)
    question_2(users_df)
    question_3(repos_df)
    question_4(users_df)
    question_5(repos_df)
    question_6(users_df, repos_df)
    question_7(repos_df)
    question_8(users_df)
    question_9(users_df)
    question_10(users_df)
    question_11(repos_df)
    question_12(users_df)
    question_13(users_df)
    question_14(repos_df)
    question_15(users_df)
    question_16(users_df)

if __name__ == "__main__":
    main()
