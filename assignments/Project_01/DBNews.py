#-*- coding=utf-8 -*-
import pymysql
import re
import jieba
import langconv 
import platform
import os

host = 'rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com'
port = 3306
user = 'root'
password = 'AI@2019@ai'
charset = 'utf8'
db = 'stu_db'
table_name = 'news_chinese'

#切词之后保存文件位置，可修改，可能需要创建这个目录
sysStr = platform.system()
if sysStr == 'Windows':
	cutFileName = 'D:\python\datasource\project/news_cut.txt'
elif sysStr == 'Linux':
	cutFileName = './datasource/news_cut.txt'

if not os.path.exists(cutFileName):
	open(cutFileName,'w').close()

	
#连接数据库，取得数据，繁简转换，清洗，切词并保存
def getDBNews():
	conn = pymysql.Connect(host=host,port=port,user=user,password=password,charset=charset,db=db)
	if not conn: return False
	cursor = conn.cursor()
	result = cursor.execute("select * from news_chinese")
	results = cursor.fetchall()
	fileNews = open(cutFileName,'w', encoding = 'utf8')

	for rs in results:
		if not rs[3]: continue
		news = rs[3].strip()
		if not news: continue
		text = "".join(re.findall(r'[^\\n]',news))
		text = "".join(re.findall(r'[\d|\w]+',text))#数据清洗
		text = langconv.Converter('zh-hans').convert(text)#繁简转换
		cut_text = list(jieba.cut(text))#切词		
		fileNews.write(" ".join(cut_text) + '\n')
	fileNews.close()    
	conn.close()
	return True

#加载切词之后的news
def getCutNews():	
	newsContent = []
	with open(cutFileName, 'r', encoding ='utf8') as f:
		for line in f:
			line = line.strip()
			if not line: continue
			newsContent.append(line)
	return newsContent

def getNews():
	print('Loading file ... ')
	newsContent = getCutNews()
	if len(newsContent) > 0:
		print('Load file Successfull ... ')
		return newsContent
	print('Load file failed, try to get data from database ...')
	if not getDBNews():
		print("Database not available")
		return None
	print('Loading file ... ')
	newsContent = getCutNews()
	print('Loading file Successfull ... ')
	return newsContent

if __name__ == '__main__':
	newsContent = getNews()	
	print('length of news:' + str(len(newsContent)))
