
# coding: utf-8

# In[1]:

import re
import pylab as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer ,CountVectorizer
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import roc_auc_score as auc
from collections import defaultdict,Counter
from tqdm import tqdm
import gc
import lightgbm as lgb
from textblob import TextBlob
import xgboost as xgb
from multiprocessing import Pool
from scipy.stats import pearsonr
from scipy.sparse import  hstack
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem  import PorterStemmer


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.family']='sans-serif'


# In[2]:


resource = pd.read_csv('data/resources.csv',low_memory=False)
train_data = pd.read_csv('data/train.csv',low_memory=False)
test_data = pd.read_csv('data/test.csv',low_memory=False)
train_y = train_data['project_is_approved']
id_test = test_data['id'].values


# In[ ]:




# In[ ]:




# # 下面进行数据清洗、特征工程

# * <font color = red> 首先进行缺失值的处理

# * project_essay_4	
# * project_essay_3	
# * description	
# * teacher_prefix	
# 
# 以上几个项存在缺失
# 首先合并test和train数据,将resource文件中特征加入

# In[3]:


train_data['is_train_data'] = 1
test_data['is_train_data'] = 0
train_X = train_data.drop('project_is_approved',axis=1,inplace=False)
test_X = test_data
combine_data = pd.concat([train_X,test_X],axis=0,ignore_index=True)


# In[4]:


#从 resource文件中产生更多特征
resource['total_price'] = resource['quantity'] * resource['price']
#产生 total_price特征
new_features = resource.groupby(['id'],as_index=False)[['total_price']].sum()
combine_data = pd.merge(combine_data,new_features,how='left',on='id')

new_features = resource.groupby(['id'],as_index=False)[['total_price']].mean()
new_features = new_features.rename(columns={'total_price':'mean_price'})
combine_data = pd.merge(combine_data,new_features,how='left',on='id')

new_features = resource.groupby(['id'],as_index=False)[['quantity']].count()
new_features = new_features.rename(columns={'quantity':'quantity_count'})
combine_data = pd.merge(combine_data,new_features,how='left',on='id')

new_features = resource.groupby(['id'],as_isendex=False)[['quantity']].sum()
new_features = new_features.rename(columns={'quantity':'quantity_sum'})
combine_data = pd.merge(combine_data,new_features,how='left',on='id')


new_features = resource.groupby(['id'],as_index = False).agg({'description':lambda x:''.join(x.values.astype(str))}).rename(
    columns = {'description':'resource_description'})
combine_data = pd.merge(combine_data,new_features,how='left',on='id')



# * <font color =red>至上一步我们已经把 resource中的特征提取完毕,接下来处理缺失值
#     essay先不处理 之后需要用到

# In[ ]:


# In[5]:


combine_data['teacher_prefix']=combine_data['teacher_prefix'].fillna(combine_data['teacher_prefix'].mode()[0])

# * <font color =red>至此完成了所有的缺失值填充，接下来考虑产出一些新特征

# # 1、通过prefix可以产出性别特征

# In[6]:


def add_gender(data):
    if data == 'Mr.':
        return 'male'
    elif  data=='Mrs.' or data=='Ms.':
        return 'female'
    else:
        return 'unknown'

combine_data ['gender'] = combine_data['teacher_prefix'].apply(add_gender)


# # 2、school state 取分析接受率的前5名进行标记

# In[ ]:


#找出前5名索引,和后5名索引
school_state_sort_index = train_data[['project_is_approved','school_state']].groupby(['school_state']).mean().sort_values(by='project_is_approved',ascending =False).head(5).index
school_state_tail_index = train_data[['project_is_approved','school_state']].groupby(['school_state']).mean().sort_values(by='project_is_approved',ascending =False).tail(5).index
combine_data['school_state_is_top_5'] = combine_data['school_state'].apply(lambda x:1 if x in school_state_sort_index else 0 )
combine_data['school_state_is_tail_5'] = combine_data['school_state'].apply(lambda x:1 if x in school_state_tail_index else 0 )


# # 3、对essay进行处理

# In[7]:


combine_data['project_essay'] = combine_data.apply(lambda row:''.join([
                             str(row['project_essay_1']),
                             str(row['project_essay_2']),
                             str(row['project_essay_3']),
                             str(row['project_essay_4'])
                             ]),axis=1)


# # 4、对project_subject_catagory 和 project_subject_subcatagory标记前五名和后五名

# In[ ]:


#找出前5名索引,和后5名索引
project_subject_catagory_head_index = train_data[['project_is_approved','project_subject_categories']].groupby(['project_subject_categories']).mean().sort_values(by='project_is_approved',ascending =False).head(5).index
project_subject_catagory_tail_index = train_data[['project_is_approved','project_subject_categories']].groupby(['project_subject_categories']).mean().sort_values(by='project_is_approved',ascending =False).tail(5).index
combine_data['project_subject_catagory_is_top_5'] = combine_data['project_subject_categories'].apply(lambda x:1 if x in project_subject_catagory_head_index else 0 )
combine_data['project_subject_catagory_is_tail_5'] = combine_data['project_subject_categories'].apply(lambda x:1 if x in project_subject_catagory_tail_index else 0 )


project_subject_subcatagory_head_index = train_data[['project_is_approved','project_subject_subcategories']].groupby(['project_subject_subcategories']).mean().sort_values(by='project_is_approved',ascending =False).head(5).index
project_subject_subcatagory_tail_index = train_data[['project_is_approved','project_subject_subcategories']].groupby(['project_subject_subcategories']).mean().sort_values(by='project_is_approved',ascending =False).tail(5).index
combine_data['project_subject_subcatagory_is_top_5'] = combine_data['project_subject_subcategories'].apply(lambda x:1 if x in project_subject_subcatagory_head_index else 0 )
combine_data['project_subject_subcatagory_is_tail_5'] = combine_data['project_subject_subcategories'].apply(lambda x:1 if x in project_subject_subcatagory_tail_index else 0 )


# # 5、提取essay中的长度特征

# In[8]:


def extract_length_from_essay(data):
    data['project_title_len'] = data['project_title'].apply(lambda x:len(str(x)))
    data['project_essay1_len'] = data['project_essay_1'].apply(lambda x:len(str(x)))
    data['project_essay2_len'] = data['project_essay_2'].apply(lambda x:len(str(x)))
    data['project_essay3_len'] = data['project_essay_3'].apply(lambda x:len(str(x)))
    data['project_essay4_len'] = data['project_essay_4'].apply(lambda x:len(str(x)))
    data['project_resource_summary_len'] = data['project_resource_summary'].apply(lambda x:len(str(x)))
    data['resource_description_len'] = data['resource_description'].apply(lambda x:len(str(x)))

    data['project_title_word_len'] = data['project_title'].apply(lambda x:len(str(x).split(' ')))
    data['project_essay1_word_len'] = data['project_essay_1'].apply(lambda x:len(str(x).split(' ')))
    data['project_essay2_word_len'] = data['project_essay_2'].apply(lambda x:len(str(x).split(' ')))
    data['project_essay3_word_len'] = data['project_essay_3'].apply(lambda x:len(str(x).split(' ')))
    data['project_essay4_word_len'] = data['project_essay_4'].apply(lambda x:len(str(x).split(' ')))
    data['project_resource_summary_word_len'] = data['project_resource_summary'].apply(lambda x:len(str(x).split(' ')))
    data['resource_description_word_len'] = data['resource_description'].apply(lambda x: len(str(x).split(' ')))
    
extract_length_from_essay(combine_data)


# # 6、处理时间特征

# In[9]:


def extract_time_features(data):
    print(u'开始处理时间特征')
    timestamp = pd.to_datetime(data['project_submitted_datetime'])
    data['year'] = timestamp.dt.year
    data['month'] = timestamp.dt.month
    data['day'] = timestamp.dt.day
    data['weekday'] = timestamp.dt.weekday
    data['hour'] = timestamp.dt.hour
    data['minute'] = timestamp.dt.minute
    data['second'] = timestamp.dt.second
    print(u'处理完成')
extract_time_features(combine_data)

#8、进行情感分析
textColumns = ['project_title','project_essay',
               'project_resource_summary','resource_description']

def getSentiment(sent):
    Text = TextBlob(sent).sentiment
    return (Text.polarity ,Text.subjectivity)

def extract_sentimental_features(data):
    print('正在提取情感信息')
    for col in textColumns:
        temp = np.array(list(map(getSentiment,data[col])))
        data[col+'_pol'] = temp[:,0]
        data[col+'_sub'] = temp[:,1]
        print('%s 情感提取完毕'%col)
    print('情感特征提取完毕')
extract_sentimental_features(combine_data)

#9、进行重点词提取、计算每篇文章出现的重点词数量
KeyChars = ['!', '\?', '@', '#', '\$', '%', '&', '\*', '\(', '\[', '\{', '\|', '-', '_', '=', '\+',
            '\.', ':', ';', ',', '/', '\\\\r', '\\\\t', '\\"', '\.\.\.', 'etc', 'http', 'poor',
            'military', 'traditional', 'charter', 'head start', 'magnet', 'year-round', 'alternative',
            'art', 'book', 'basics', 'computer', 'laptop', 'tablet', 'kit', 'game', 'seat',
            'food', 'cloth', 'hygiene', 'instraction', 'technolog', 'lab', 'equipment',
            'music', 'instrument', 'nook', 'desk', 'storage', 'sport', 'exercise', 'trip', 'visitor',
            'my students', 'our students', 'my class', 'our class']

def extract_keychar_features(data):
    print('正在提取重点词汇....')
    for col in textColumns:
        for c in KeyChars:
            data[col+'_'+c] = data[col].apply(lambda x:len(re.findall(c,x.lower())))
            print('%s 关键词: %s 提取完毕'%(col,c))
    print('关键词特征提取完毕')
extract_keychar_features(combine_data)

#10、进行公共词提取步骤
def extract_common_word_features(data):
    print('开始提取公共词特征....')
    for i,col1 in enumerate(textColumns[:-1]):
        for col2 in textColumns[i+1:]:
            data['%s_%s_common'%(col1,col2)] = \
                data.apply(lambda row:len(set(re.split('\W',row[col1].lower())).intersection(re.split('\W',row[col2].lower()))),axis = 1)
            print('%s 和 %s 公共词寻找完毕'%(col1,col2))
    print('公共词特征提取完毕')
extract_common_word_features(combine_data)

#11、加入一些额外的多项式特征
#提取出是数字的特征
numeric_feature_index = ['teacher_number_of_previously_posted_projects',
       'total_price', 'mean_price', 'quantity_count', 'quantity_sum',
       'project_title_len',
       'project_essay1_len', 'project_essay2_len', 'project_essay3_len',
       'project_essay4_len', 'project_resource_summary_len',
       'resource_description_len', 'project_title_word_len',
       'project_essay1_word_len', 'project_essay2_word_len',
       'project_essay3_word_len', 'project_essay4_word_len',
       'project_resource_summary_word_len', 'resource_description_word_len',
        ]

#提取出训练数据
Train_numeric = combine_data.loc[combine_data.is_train_data==1,:][numeric_feature_index]
cv = 10
index = [np.random.randint(0,Train_numeric.shape[0],int(Train_numeric.shape[0]/cv))\
         for k in range(cv)]

#10次交叉验证
Corr = {}

for c in numeric_feature_index:
    C1 ,P1 = np.nanmean([pearsonr(train_y[index[k]],(1+Train_numeric[c].iloc[index[k]]))
                         for k in range(cv)],axis=0)
    C2 ,P2 = np.nanmean([pearsonr(train_y[index[k]],1/(1+Train_numeric[c].iloc[index[k]]))
                         for k in range(cv)],axis=0)
    if P2<P1:
        combine_data[c] =1/(1+ combine_data[c])
        Corr[c] = [C2,P2]
    else:
        combine_data[c] = combine_data[c]+1
        Corr[c] = [C1,P1]

polyCol = []
thrP =0.01
thrC =0.02

for i,c1 in enumerate(numeric_feature_index[:-1]):
    C1,P1 = Corr[c1]
    for c2 in numeric_feature_index[i+1:]:
        C2,P2 =Corr[c2]
        V = Train_numeric[c1]*Train_numeric[c2].values
        C,P =np.nanmean([pearsonr(train_y[index[k]],V[index[k]]) for k in range(cv)],axis=0)
        if P<thrP and abs(C) - max(abs(C1),abs(C2)) >thrC:
            combine_data[c1+'_'+c2+'_poly'] = combine_data[c1]*combine_data[c2]
            polyCol.append(c1+'_'+c2+'_poly')
            print(c1 + '_' + c2, '\t\t(%g, %g)\t(%g, %g)\t(%g, %g)' % (C1, P1, C2, P2, C, P))

del Train_numeric
gc.collect()

# In[10]:


gc.collect()
cols = ['project_title','project_essay','project_resource_summary']
n_features = [400,1000,400]

print(u'开始进行文本处理')
for c_i,c in tqdm(enumerate(cols)):
    tfidf = TfidfVectorizer(max_features=n_features[c_i],norm='l2')
    tfidf.fit(combine_data[c])
    tfidf_combine = np.array(tfidf.transform(combine_data[c]).toarray(),dtype=np.float16)
    
    for j in range(n_features[c_i]):
        combine_data[c+'_tfidf_'+str(j)] = tfidf_combine[:,j]
    del tfidf,tfidf_combine
    gc.collect()
print(u'完成文本特征提取')


# # 8、 下面进行一些catagory特征的编码
def extract_catagory_features(data,cat):
    vectorizer = CountVectorizer(binary=True,
                                 ngram_range=(1,1),
                                 tokenizer=lambda x :[word.strip() for word in x.split(',')])
    for i,col in enumerate(cat):
        vec = vectorizer.fit_transform(data[col].fillna(''))
        if i==0:
            cat_features = vec
        else:
            cat_features = hstack((cat_features ,vec))
        del vec
        gc.collect()
    return cat_features

#此时的输出为稀疏矩阵
cat_feat = ['project_subject_categories','project_subject_subcategories']
cat_features = extract_catagory_features\
    (combine_data,cat_feat)
cat_features = cat_features.toarray()
cat_features = pd.DataFrame(cat_features)
# In[11]:


catagory_feature = [
    'teacher_id',
    'teacher_prefix',
    'gender',
    'school_state',
    'project_grade_category'
]
print(u'开始编码')
for i,col in enumerate(catagory_feature):
    encoder = LabelEncoder()
    encoder.fit(combine_data[col].astype(str))
    combine_data[col] = encoder.transform(combine_data[col].astype(str))
    del encoder
    gc.collect()
print(u'完成编码')


# # 9、最后丢掉一些冗余特征

# In[12]:


drop_columns = [
    'id',
    'project_submitted_datetime',
    'project_title',
    'project_essay_1', 
    'project_essay_2',
     'project_essay_3', 
    'project_essay_4', 
    'project_resource_summary',
    'project_essay',
    'project_subject_categories',
    'project_subject_subcategories',
    'resource_description'
]

combine_data = pd.concat([combine_data,cat_features],axis=1)
combine_data = combine_data.drop(labels=drop_columns,axis=1,errors='ignore')
float64_index = combine_data.select_dtypes(include=np.float64).columns
combine_data[float64_index] = combine_data[float64_index].astype(np.float16)


# In[13]:


#分离训练集和测试集

train_X = combine_data.loc[combine_data.is_train_data ==1,:].drop('is_train_data',axis=1,errors='ignore')
test_X = combine_data.loc[combine_data.is_train_data != 1 ,:].drop('is_train_data',axis=1,errors='ignore')

del train_data,test_data,resource,combine_data,cat_features
gc.collect()


# # 至以上步骤我们已经完成了特征工程，下面进行模型搭建

# In[15]:


#建立模型

feature_names = list(train_X.columns)
cnt = 0 
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=0)
auc_buf=[]

for train_index,valid_index in kf.split(train_X):
    print('Fold {}/{}'.format(cnt+1,n_splits))
    lgb_params ={
        'boosting_type': 'gbdt',
        'objective':'binary',
        'metric':'auc',
        'max_depth':14,
        'num_leaves':31,
        'learning_rate':0.025,
        'feature_fraction':0.85,
        'bagging_fraction':0.85,
        'bagging_freq':5,
        'verbose':0,
        'lambda_l2':1.0,
        'num_threads': 1,
        'min_gain_to_split':0,
    }
    lgb_train = lgb.Dataset(
    train_X.loc[train_index],
    train_y.loc[train_index],
    feature_name=feature_names)
    
    lgb_train.raw_data = None
    
    lgb_valid = lgb.Dataset(
    train_X.loc[valid_index],
    train_y.loc[valid_index])
    
    lgb_valid.raw_data= None
    
    lgb_clf = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=10000,
    valid_sets=[lgb_train,lgb_valid],
    early_stopping_rounds=100,
    verbose_eval=100
    )
    
    if cnt==0:
        importance = lgb_clf.feature_importance()
        model_fnames = lgb_clf.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1]>0]
        print('Important features')
        for i in range(60):
            if i< len(tuples):
                print(tuples[i])
            else:
                break
        del importance,model_fnames,tuples
        gc.collect()
    
    p = lgb_clf.predict(train_X.loc[valid_index],num_iteration = lgb_clf.best_iteration)
    auc = roc_auc_score(train_y.loc[valid_index],p)
    
    print('{} AUC: {}'.format(cnt,auc))
    
    p = lgb_clf.predict(test_X,num_iteration = lgb_clf.best_iteration )
    if len(p_buf) == 0:
        p_buf = np.array(p,dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    auc_buf.append(auc)
    
    cnt += 1
    
#     if cnt>0:
#         break
        
    del lgb_clf,lgb_train,lgb_valid,p
    gc.collect()

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean,auc_std))

preds = p_buf/cnt

# Prepare submission
subm = pd.DataFrame()
subm['id'] = id_test
subm['project_is_approved'] = preds
subm.to_csv('data/submission.csv', index=False,sep=',')
