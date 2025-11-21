import streamlit as st
import requests
from bs4 import BeautifulSoup
import time

# 제목
st.title("네이버 검색 결과 크롤링")

# 검색어 입력
query = st.text_input("검색어를 입력하세요")

if query:
    # URL
    url = f"https://search.naver.com/search.naver?query={query}"

    # 헤더 설정 (User-Agent 필수)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    }

    # requests모듈의 get()함수를 이용하여 웹크롤링, urlopen(url)과 비슷
    response = requests.get(url, headers=headers)    
    time.sleep(1)
    if response.status_code == 200: #정상적으로 웹페이지를 받았을때 코드가 200임
        soup = BeautifulSoup(response.text, "html.parser")

        st.subheader(f"'{query}' 검색 결과")

        find_items = soup.find_all('여기에 검색할 태그와 속성을 입력해주세요.')
        if find_items:
            st.write(f"{len(find_items)}개 찾았습니다.")
            for item in find_items:
                # 아래 item.span.text 부분은 해당 태그를 확인하여 텍스트를 추출
                st.write(item.span.text, item.a['href'])
        else:
            st.write("검색 결과가 없습니다.")
    else:
        st.error("검색에 실패했습니다.")
