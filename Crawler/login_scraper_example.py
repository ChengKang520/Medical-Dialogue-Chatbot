
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,import-outside-toplevel,invalid-name
import selenium
from selenium import webdriver
import requests
import re
from bs4 import BeautifulSoup
from time import sleep
import io, json

print('Selenium version is {}'.format(selenium.__version__))
print('-' * 80)

def main():
    #-----------------------------------------------------------------------------
    config = {
        "url": {
          "login": "https://passport.haodf.com/user/showlogin",
          "target": "https://www.haodf.com/bingcheng/8822504854.html"  # https://www.haodf.com/kanbing/6478777497.html
        },
        "account": {
          "username": "userID",
          "password": "password"
        }
    }
    login_url = config['url']['login']
    target_url = config['url']['target']
    username = config['account']['username']
    password = config['account']['password']

    #-----------------------------------------------------------------------------
    # automated testing
    print('open login page ...')
    # 这里改为你的chromedriver地址
    ChromeDriverPath = "C:/Program Files (x86)/Google/Chrome/Application/chromedriver"
    driver = webdriver.Chrome(executable_path=ChromeDriverPath)
    driver.get(login_url)

    driver.find_element_by_xpath("/html/body/div[3]/div[2]/div[2]/div/div/div[1]/ul/li[3]").click()
    driver.find_element_by_xpath("/html/body/div[3]/div[2]/div[2]/div/div/div[2]/form[2]/div/div[1]/div/input").send_keys(username)
    driver.find_element_by_xpath("/html/body/div[3]/div[2]/div[2]/div/div/div[2]/form[2]/div/div[3]/div/input").send_keys(password)
    if driver.find_element_by_xpath("/html/body/div[3]/div[2]/div[2]/div/div/div[2]/form[2]/div/div[6]/div[1]").is_selected() is False:
        driver.find_element_by_xpath("/html/body/div[3]/div[2]/div[2]/div/div/div[2]/form[2]/div/div[6]/div[1]").click()
    # 获取到登录按钮的driver后，使用click()函数，模拟点击button
    driver.find_element_by_xpath("/html/body/div[3]/div[2]/div[2]/div/div/div[2]/form[2]/div/div[9]/div[2]/a").click()

    #-----------------------------------------------------------------------------
    requests_session = requests.Session()
    print('open target page ...')
    driver.get(target_url)
    selenium_user_agent = driver.execute_script("return navigator.userAgent;")
    requests_session.headers.update({"user-agent": selenium_user_agent})
    for cookie in driver.get_cookies():
        requests_session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

    sleep(float(5))
    try:
        resp_KanBing = requests_session.get(target_url)
    except:
        print('Encoded with protect strategy!')

    soup_KanBing = BeautifulSoup(resp_KanBing.text, 'html.parser')
    KanBing_pageUrl = soup_KanBing.select('[class=m-b-content]')
    BingCheng_pageUrl = KanBing_pageUrl[0]['href']
    driver.get(BingCheng_pageUrl)
    sleep(float(5))
    while 1:
        try:
            sleep(float(0.1))
            driver.find_element_by_xpath("/html/body/main/section[1]/section/section[3]/div[3]/div[2]/span").click()
            # /html/body/main/section[1]/section/section[3]/div[3]/div[2]/span
        except:
            break
    resp_BingCheng = requests_session.get(BingCheng_pageUrl)
    soup_Bingcheng = BeautifulSoup(resp_BingCheng.text, 'html.parser')

# /html/body/main/section[1]/section/section[3]/div[2]/div[15]  .get_text()
    # go to the diseaseinfo box
    con = soup_Bingcheng.find_all(id='app')  # 获取每一个填写的div块
    con_diseaseinfotitle = con[0].find('h2', class_="diseaseinfotitle").get_text()
    info3_title_list = con[0].find_all('div', class_="info3-title ")  # 找到文章列表
    info3_value_list = con[0].find_all('div', class_="info3-value ")
    for i in range(len(info3_title_list)):
        print(str(info3_title_list[i].get_text() + info3_value_list[i].get_text()))


    # go to the suggestions box
    con_suggestions= con[0].find('h2', class_="suggestions-maintitle").get_text()
    suggestions_text_list = con[0].find_all('div', class_="curr-head-wrap")  # 找到文章列表
    suggestions_content_list = con[0].find_all('p', class_="suggestions-text-value")
    for i in range(len(suggestions_text_list)):
        print(str(suggestions_text_list[i].get_text() + suggestions_content_list[i].get_text()))


    # go to the msgboard box
    con_msg = con[0].find('h2', class_="msgtitle").get_text()
    # msg_him_list = con[0].find_all('div', class_="content-him content-text ")  # 找到文章列表
    msg_list = con[0].find_all('div', class_="m-b-content")
    flag_before = False  # False is doctor; True is patient
    flag_after = False
    flag_current = False
    msg_doctor = '\n'
    msg_patient = '\n'
    msg_content = []
    msg_all = []
    for i in range(len(msg_list)):
        if (i == 0) & (msg_list[i].find('div', class_="content-him content-text ") is None):
            flag_current = True   #True is patient;
            msg_patient = msg_list[i].find('div', class_="content-him content-text content-him-patient").get_text()

        if (i == 0) & (msg_list[i].find('div', class_="content-him content-text content-him-patient") is None):
            flag_current = False   #False is doctor;
            msg_doctor = msg_list[i].find('div', class_="content-him content-text ").get_text()

        flag_before = flag_current

        try:
            if (i > 0):
                msg_doctor = msg_doctor + msg_list[i].find('div', class_="content-him content-text ").get_text()
                flag_current = False
        except:
            print('this is not doctor!')
            # flag_current = False

        try:
            if (i > 0):
                msg_patient = msg_patient + msg_list[i].find('div', class_="content-him content-text content-him-patient").get_text()
                flag_current = True
        except:
            print('this is not patient!')
            # flag_patient = False

        # ha = 1

        if flag_before != flag_current:
            if flag_before is False:  #False is doctor;
                # msg_doctor_list.append({'doctor':msg_doctor})
                msg_content.append({'doctor': msg_doctor})
                msg_doctor = []
            if flag_before is True:  #True is patient;
                # msg_patient_list.append(msg_patient)
                msg_content.append({'patient': msg_patient})
                msg_patient = []

        if i == len(msg_list)-1:
            if flag_current is False:  #False is doctor;
                # msg_doctor_list.append({'doctor':msg_doctor})
                msg_content.append({'doctor': msg_doctor})
                msg_doctor = []
            if flag_current is True:  #True is patient;
                # msg_patient_list.append(msg_patient)
                msg_content.append({'patient': msg_patient})
                msg_patient = []

    with io.open('data.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(msg_content, ensure_ascii=False))


if __name__ == "__main__":
    main()

