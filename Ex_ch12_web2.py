import streamlit as st
import requests
from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import time
from matplotlib import rc, font_manager

font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 기준
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 페이지 제목
st.title("뉴스 트렌드 대시보드")
search = st.button("검색")

if search:
    st.write(f"검색 중...")

    # 네이버 뉴스 검색 URL
    url = f"https://news.naver.com/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    time.sleep(1) 

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # 뉴스 제목 추출
        news_items = soup.find_all(class_='cnf_news _cds_link _editn_link')
        #news_items = soup.select("a.news_tit")
        st.write(f"{len(news_items)}개 찾았습니다.")

        titles = [item.text for item in news_items[:20]]  # 상위 20개

        if titles:
            st.subheader("뉴스 기사 제목")
            for t in titles:
                st.write(f"- {t}")

            # 키워드 분석 (간단히 단어 빈도)
            words = []
            for t in titles:
                words += re.findall(r'[A-Za-z0-9]+|[가-힣]+', t)  # 단어 단위로 분리

            counter = Counter(words)
            most_common = dict(counter.most_common(50))  # 상위 50개
            # 워드클라우드 생성
            wc = WordCloud(
                width=800, 
                height=400, 
                font_path=font_path,
                background_color="white")

            wc.generate_from_frequencies(most_common)
            plt.figure(figsize=(8, 5)) 
            words_list = [w for w, _ in most_common.items()] 
            counts_list = [c for _, c in most_common.items()]

            # 막대그래프
            fig_bar, ax = plt.subplots(figsize=(8,5))
            ax.barh(words_list[:10:-1], counts_list[:10:-1])
            ax.set_title("상위 단어 빈도수", fontsize=14)
            ax.set_xlabel("빈도", fontsize=12)
            st.pyplot(fig_bar)

            # 워드클라우드
            fig_wc, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.write("뉴스 기사가 없습니다.")
    else:
        st.error("뉴스 검색 실패")
