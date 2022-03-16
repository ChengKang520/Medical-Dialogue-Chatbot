#-*-coding:UTF-8-*-
#-*-encoding=UTF-8-*-

from bs4 import BeautifulSoup
from time import sleep
import re
import requests
import traceback

# Global Variables

wait = 1
defaultHeaders = {
    'User-Agent': ''
}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'
}
defaultSeperator = ','

sourceFilePath = './step-1-result.csv'
resultFilePath = './step-2-result.csv'

print('Starting crawling more doctor links ...')

counter = 0

with open(sourceFilePath) as sf:

    # skip header line
    next(sf)

    with open(resultFilePath, 'w') as rf:

        rf.write(defaultSeperator.join(['医生姓名','医生ID','个人网站','评价分享链接','医院科室','信息中心页','地区']))

        for row in sf:

            # [n,d,l,p] ~= [医生姓名,医院科室,信息中心页,地区]
            [n,d,l,p] = row.strip().split(defaultSeperator)
            counter+=1
            print('No '+str(counter)+', Crawling more links of '+n+', '+l)

            try:
                personalWeb = ''
                doctorID = ''
                offlineCommentLink = ''

                # CRAWL OFFLINE COMMENT LINKS
                # jingyanLink = l.replace('.htm','/jingyan/1.htm')
                # offlineCommentLink = 'https:'+requests.head(jingyanLink, headers=defaultHeaders).headers.get('Location')

                # CRAWL PERSONAL WEBSITE LINKS
                sleep(float(wait))
                res = requests.get(l[6:], headers = defaultHeaders)

                # CRAWL DOCTOR ID
                print('*****************************************')
                ID_address = l.split('/')
                ID_address = ID_address[-1]
                doctorID = ID_address[:-5]
                rf.write('\n'+defaultSeperator.join([n,doctorID,personalWeb,offlineCommentLink,d,l,p]))

            except Exception as e:
                traceback.print_exc()
                exit(1)

print('Finished crawling more doctor links.\n')
