# 네이버 블로그 검색 api python
import urllib.request
import json
import re
import pandas as pd

# selenium 동적 크롤링 python
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


CLIENT_ID = "MbuEYulgGLLeJwRW2ugN"
CLIENT_SECRET = "_igAnMpGRM"

def get_title_link(df, search, CLIENT_ID, CLIENT_SECRET, start=1):
    encText = urllib.parse.quote(search)
    params = {
        'query': encText,   # 검색 문자열
        'display': 100,      # 출력 건수. (기본) 10 (최대) 100
        'start': start,         # 검색 시작 위치. (기본) 1 (최대) 1000
                }
    params_text = "&".join([key +"=" + str(params[key]) for key in params.keys()])
    url = "https://openapi.naver.com/v1/search/blog?" +  params_text # json 결과


    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)

    response = urllib.request.urlopen(request)
    rescode = response.getcode()

    if rescode == 200:
        response_body = response.read()
        decode = response_body.decode('utf-8')
        response_dict = json.loads(decode)
        items = response_dict['items']
        remove_tag = re.compile('<.*?>')
        for item in items:
            title = re.sub(remove_tag, '', item['title'])
            link = item['link']
            description = re.sub(remove_tag, '', item['description'])
            new_data = {
                'title': [title],
                'link': [link],
                'description': [description]
            }
            new_df = pd.DataFrame(new_data)
            if not df.empty:
                df = pd.concat([df, new_df])
            else:
                df = new_df
    else:
        print("Error Code: " + rescode)
    return df
# search(CLIENT_ID, CLIENT_SECRET)


def get_description(df):
    contents = []
    nan_index = df[df['description'].isnull()].index
    url_lst = list(df['link'][nan_index])
    # 크롬 드라이버 설치
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(5)
    for url in url_lst:
        driver.get(url)
        time.sleep(1)
        driver.switch_to.frame('mainFrame')
        try:
            a = driver.find_element(By.CSS_SELECTOR, 'div.se-main-container').text
            a = re.sub('\n', ' ', a)
            contents.append(a)
        except NoSuchElementException:
            a = driver.find_element(By.CSS_SELECTOR, 'div#content-area').text
            a = re.sub('\n', ' ', a)
            contents.append(a)
    print(len(contents))
    df['description'][nan_index] = pd.Series(contents)
    driver.quit()
    return df