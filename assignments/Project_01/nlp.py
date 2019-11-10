import re
import os
LTP_DATA_DIR = '/python/datasource/ltp_data_v3.4.0'  # ltp模型目录的路径

from pyltp import SentenceSplitter

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型

pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
from pyltp import Postagger
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型

ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型

par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
from pyltp import Parser
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型
postags = ['nh', 'r', 'r', 'v']
key_word = set(['说','表示', '告诉', '认为', '指出', '坦言', '看来', '称', '介绍',  '透露', '强调',  '写道', '话', '建议',  '问', '提到', '声称', '称赞', '深有体会', '道', '设想', '直言', '解释', '称', '相信', '呼吁', '担心', '写信给', '深知', '在我看来', '证实', '眼中', '宣称', '清楚', '还称', '回答',  '估计', '来说','承认', '觉得', '特别强调',  '毫不讳言',  '坚信', '提及', '非常重视', '谈到', '申明', '感慨', '普遍认为','表明', '看法', '援引', '供词',  '知晓',  '批评','想法',  '描述', '重申','告诫', '指责',  '坦承','李锦说', '如是说', '敦促', '表态', '断定', '理解', '坚称',  '提醒'])

def cut(text):
    cut_text = []
    for t in text:
        t = t.strip()
        if not t: continue                
        words = segmentor.segment(t)
        cut_text.append(' '.join(words))
    return cut_text
def sentence_spliter(text):
    sents = []
    for s in SentenceSplitter.split(text):
        s = s.strip()
        if not s: continue
        sents.append(s)
    return sents

class NewsExtract():
    def __init__(self):        
        self.text = []
        self.sentences = []
        self.cut_text = []
        self.peoples = []
        self.viewpoints = {}
    def preprocessing(self, text):
        text = text.strip()
        if not text: 
            self.text = []
            self.sentences = []
            self.cut_text = []
            return 
        self.text = text  
        self.sentences = sentence_spliter(self.text)
        self.cut_text = cut(self.sentences)
        self.get_ner()
    def get_ner(self):
        self.peoples = []
        for i, s in enumerate(self.cut_text):
            s = s.split()
            sen_set = set(s)
            inter_set = sen_set & key_word
            if not inter_set: continue
            print(inter_set)
            postags = postagger.postag(s)
            netags = ' '.join(recognizer.recognize(s, postags))
            for tag, word in zip(netags.split(), s):
                if tag == 'S-Nh':
                    self.peoples.append(word)
                    if word not in self.viewpoints: self.viewpoints[word] = []
                    self.viewpoints[word].append((i,self.sentences[i]))    
    def get_viewpoints(self):        
        return self.viewpoints
    
    
if __name__ == '__main__':
    sentence='''据巴西《环球报》7日报道，巴西总统博索纳罗当天签署行政法令，放宽枪支进口限制，并增加民众可购买弹药的数量。\r\n《环球报》称，该法令最初的目的是放松对收藏家与猎人的限制，但现在扩大到其他条款。新法令将普通公民购买枪支的弹药数量上限提高至每年5000发，此前这一上限是每年50发。博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”。另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”\r\n这不是巴西第一次放宽枪支限制。今年1月，博索纳罗上台后第15天就签署了放宽公民持枪的法令。根据该法令，希望拥有枪支的公民须向联邦警察提交申请，通过审核者可以在其住宅内装备最多4把枪支，枪支登记有效期由5年延长到10年。《环球报》称，博索纳罗在1月的电视讲话中称，要让“好人”更容易持有枪支。“人民希望购买武器和弹药，现在我们不能对人民想要的东西说不”。\r\n2004年，巴西政府曾颁布禁枪法令，但由于多数民众反对，禁令被次年的全民公投否决。博索纳罗在参加总统竞选时就表示，要进一步放开枪支持有和携带条件。他认为，放宽枪支管制，目的是为了“威慑猖狂的犯罪行为”。资料显示，2017年，巴西发生约6.4万起谋杀案，几乎每10万居民中就有31人被杀。是全球除战争地区外最危险的国家之一。\r\n不过，“以枪制暴”的政策引发不少争议。巴西《圣保罗页报》称，根据巴西民调机构Datafolha此前发布的一项调查，61%的受访者认为应该禁止持有枪支。巴西应用经济研究所研究员塞奎拉称，枪支供应增加1%，将使谋杀率提高2%。1月底，巴西民众集体向圣保罗联邦法院提出诉讼，质疑博索纳罗签署的放宽枪支管制法令。\r\n巴西新闻网站“Exame”称，博索纳罗7日签署的法案同样受到不少批评。公共安全专家萨博称，新的法令扩大了少数人的特权，不利于保护整个社会。（向南）\r\n'''
    sentence = ''.join(re.findall(r'[^\\n]', sentence))
    news_extract = NewsExtract()
    news_extract.preprocessing(sentence)
    view = news_extract.get_viewpoints()
    for k in view:
        print(k, view[k])
    
    
    
    