import requests
import pandas as pd
import time
import os
from datetime import datetime

# Your GitHub Personal Access Token
TOKEN = 'xxx'

# Headers for authentication
headers = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
}

search_query = 'location:zurich followers:>50'
per_page = 100  # Maximum per page

users_list = []
page = 1
while True:
    print(f"Fetching page {page} of users...")
    params = {
        'q': search_query,
        'per_page': per_page,
        'page': page,
    }
    response = requests.get('https://api.github.com/search/users', headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error fetching users: {response.status_code}")
        break
    data = response.json()
    users = data.get('items', [])
    if not users:
        break
    users_list.extend(users)
    # Check if we've reached the last page
    if 'next' not in response.links:
        break
    page += 1
    time.sleep(1)  # Respect API rate limits

user_details_list = []

for user in users_list:
    login = user['login']
    print(f"Fetching data for user: {login}")
    user_url = f"https://api.github.com/users/{login}"
    response = requests.get(user_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching user {login}: {response.status_code}")
        continue
    user_data = response.json()
    user_details = {
        'login': user_data.get('login', ''),
        'name': user_data.get('name', ''),
        'company': user_data.get('company', ''),
        'location': user_data.get('location', ''),
        'email': user_data.get('email', ''),
        'hireable': user_data.get('hireable', ''),
        'bio': user_data.get('bio', ''),
        'public_repos': user_data.get('public_repos', 0),
        'followers': user_data.get('followers', 0),
        'following': user_data.get('following', 0),
        'created_at': user_data.get('created_at', ''),
    }
    user_details_list.append(user_details)
    time.sleep(1)  # Respect API rate limits

repositories_list = []

for user in user_details_list:
    login = user['login']
    print(f"Fetching repositories for user: {login}")
    page = 1
    repos_collected = 0
    while repos_collected < 500:
        params = {
            'per_page': 100,
            'page': page,
            'sort': 'pushed',
            'direction': 'desc',
        }
        repos_url = f"https://api.github.com/users/{login}/repos"
        response = requests.get(repos_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching repos for user {login}: {response.status_code}")
            break
        repos = response.json()
        if not repos:
            break
        for repo in repos:
            repo_details = {
                'login': login,
                'full_name': repo.get('name', ''),
                'created_at': repo.get('created_at', ''),
                'stargazers_count': repo.get('stargazers_count', 0),
                'watchers_count': repo.get('watchers_count', 0),
                'language': repo.get('language', ''),
                'has_projects': 'true' if repo.get('has_projects', False) else 'false',
                'has_wiki': 'true' if repo.get('has_wiki', False) else 'false',
                'license_name': repo['license']['key'] if repo.get('license') else '',
            }
            repositories_list.append(repo_details)
            repos_collected += 1
            if repos_collected >= 500:
                break
        # Check if we've reached the last page
        if 'next' not in response.links:
            break
        page += 1
        time.sleep(1)  # Respect API rate limits

users_df = pd.DataFrame(user_details_list)
repos_df = pd.DataFrame(repositories_list)

# Clean company names
def clean_company_name(company):
    if company:
        company = company.strip()
        if company.startswith('@'):
            company = company[1:]
        company = company.upper()
    else:
        company = ''
    return company

users_df['company'] = users_df['company'].apply(clean_company_name)

# Convert booleans and handle nulls
users_df['hireable'] = users_df['hireable'].apply(lambda x: 'true' if x else ('false' if x is not None else ''))
users_df = users_df.fillna('')

repos_df = repos_df.fillna('')
repos_df['has_projects'] = repos_df['has_projects'].apply(lambda x: 'true' if x == 'true' else 'false')
repos_df['has_wiki'] = repos_df['has_wiki'].apply(lambda x: 'true' if x == 'true' else 'false')

# Save to CSV
users_df.to_csv('users.csv', index=False)
repos_df.to_csv('repositories.csv', index=False)
