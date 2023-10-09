import pandas as pd
import numpy as np
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
%matplotlib inline

# LOad or Reading the file
data = pd.read_csv(r'C:\Users\Vipin\Desktop\XYZCorp_LendingData.txt', sep = '\t',na_values = 'NaN',low_memory = False)

#understanding the data

data.describe()
data.head()
data.shape

#checking to see if which column has how many null values
# Graph of NA values which  is more than 80%
NA_col = data.isnull().sum()
NA_col = NA_col[NA_col.values >(0.8*len(data))]
plt.figure(figsize=(20,4))
NA_col.plot(kind='bar')
plt.title('List of Columns & NA counts where NA values are more than 80%')
plt.show()

data.isnull().sum() 

#filtering data
# Delete those columns where the data is missing more than 80%

del_columns= ['member_id','pymnt_plan','title','emp_title','sub_grade','addr_state','earliest_cr_line','zip_code','desc','last_pymnt_d','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med','mths_since_last_major_derog','policy_code','application_type','annual_inc_joint','dti_joint','verification_status_joint','open_acc_6m','open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','inq_fi','max_bal_bc','all_util','total_cu_tl','inq_last_12m','mths_since_last_record','mths_since_last_delinq']
data1 = data.drop(labels = del_columns,axis = 1)

#we can also  remove some more columns which have no need

del_columns2 = ['id','initial_list_status','out_prncp','out_prncp_inv','total_pymnt_inv']
data1 = data1.drop(labels = del_columns2,axis = 1)

data1.shape
data1.isnull().sum() 

# EDA(Exploratory Data Analysis)

# Univariate Analysis
data1['default_ind'].value_counts()
data1['default_ind'].value_counts(normalize = True)
data1['default_ind'].value_counts().plot.bar()
data1['term'].value_counts().plot.bar(title='Distribution of Brow')

(data1.default_ind.value_counts()*100)/len(data1)

# Remove rows where home_ownership'=='OTHER', 'NONE', 'ANY'
rem = ['OTHER', 'NONE', 'ANY']
data1.drop(data1[data1['home_ownership'].isin(rem)].index,inplace=True)
data1.home_ownership.unique()
data1['home_ownership'].value_counts().plot.bar()
data1['grade'].value_counts().plot.bar((title='No of borrowers'))
data1['emp_length'].value_counts().plot.bar(title='NO. of borrowers')
data1['verification_status'].value_counts().plot.bar()







# Numerical variables
# histogram
plt.hist(data['int_rate'], bins=30)
plt.title('Distribution of Interest Rates')
plt.xlabel("Interest Rates")
plt.show()

plt.hist(data['loan_amnt'], bins=15, color='red')
plt.title('Distribution of Loan Amounts')
plt.xlabel("Loan Amounts")
plt.show()

plt.hist(data['installment'], bins=15, color='green')
plt.title('Distribution of Installments')
plt.xlabel("Installments")
plt.show()



#Density plot

plt.figure(1)
plt.subplot(121)
sns.distplot(data1[ 'loan_amnt'])
sns.distplot(data1[ 'int_rate'])
sns.distplot(data1[ 'installment'])
sns.distplot(data1[ 'annual_inc'])

data1['annual_inc'].plot.box()

# remove outliers from annual income
q = data1["annual_inc"].quantile(0.995)
data1 = data1[data1["annual_inc"] < q]
data1["annual_inc"].describe()
data1['annual_inc'].plot.box()
sns.distplot(data1[ 'annual_inc'])



def univariate(df,col,vartype,hue =None):
    
    '''
    Univariate function will plot the graphs based on the parameters.
    df      : dataframe name
    col     : Column name
    vartype : variable type : continuos or categorical
                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.
                Categorical(1) : Countplot will be plotted.
    hue     : It's only applicable for categorical analysis.
    
    '''
    sns.set(style="darkgrid")
    
    if vartype == 0:
        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,8))
        ax[0].set_title("Distribution Plot")
        sns.distplot(df[col],ax=ax[0])
        ax[1].set_title("Violin Plot")
        sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")
        ax[2].set_title("Box Plot")
        sns.boxplot(data =df, x=col,ax=ax[2],orient='v')
    
    if vartype == 1:
        temp = pd.Series(data = hue)
        fig, ax = plt.subplots()
        width = len(df[col].unique()) + 6 + 4*len(temp.unique())
        fig.set_size_inches(width , 7)
        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue) 
        if len(temp.unique()) > 0:
            for p in ax.patches:
                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(loan))), (p.get_x()+0.05, p.get_height()+20))  
        else:
            for p in ax.patches:
                ax.annotate(p.get_height(), (p.get_x()+0.32, p.get_height()+20)) 
        del temp
    else:
        exit
    plt.show()
    
univariate(df=data1,col='purpose',vartype=1,hue='default_ind')   
univariate(df=data1,col='home_ownership',vartype=1,hue='default_ind')
univariate(df=data1,col='term',vartype=1,hue='default_ind')

# Bivariate Analysis

#Purpose of Loan vs Loan Amount for each default_ind

plt.figure(figsize=(16,12))
plt.title('Purpose of Loan vs Loan Amount')
sns.boxplot(data =data1, x='purpose', y='loan_amnt', hue ='default_ind')
plt.show()
 
# Correlation Matrix : All Continuos(Numeric) Variables   

data1_correlation = data1.corr()
data1_correlation   

# HeatMap: All continuos variables

ax = plt.subplots(figsize=(14, 9))
sns.heatmap(data1_correlation, 
            xticklabels=data1_correlation.columns.values,
            yticklabels=data1_correlation.columns.values,annot= True)
plt.show()
data1.shape
data1.isnull().sum()

#missing value treatment

#replacing numerical values with mean
mean = data1[['total_rev_hi_lim','tot_cur_bal','tot_coll_amt','revol_util']].mean()

data1[['total_rev_hi_lim','tot_cur_bal','tot_coll_amt','revol_util']]= data1[['total_rev_hi_lim',
     'tot_cur_bal','tot_coll_amt','revol_util']].fillna(mean)

data1.isnull().sum()

# creating a dummay variables of catagorical columns
#create dummy variables for the column
dummies = pd.get_dummies(data1['term'])
dummies = pd.get_dummies(data1['grade'])
dummies = pd.get_dummies(data1['home_ownership'])
dummies = pd.get_dummies(data1[ 'purpose'])
dummies = pd.get_dummies(data1['verification_status'])
dummies = pd.get_dummies(data1['emp_length'])
#drop the original column
data1 = data1.drop('term', axis=1)
data1 = data1.drop('grade',axis =1)
data1 = data1.drop('home_ownership',axis=1)
data1 = data1.drop( 'purpose',axis=1)
data1 = data1.drop('verification_status',axis=1)
data1 = data1.drop('emp_length',axis=1)
#add dummy variables
data1 = data1.join(dummies)
data1 = data1.join(dummies)
data1 = data1.join(dummies)
data1 = data1.join(dummies)
data1 = data1.join(dummies)
data1 = data1.join(dummies)

#Splitting the data

data2 = data1

data2['str_split'] = data2.issue_d.str.split('-')
data2['issue'] = data2.str_split.str.get(0) 
data2['d'] = data2.str_split.str.get(1)

data2['issue'].unique()
data2['d'].unique()

data2['issue'] = data2['issue'].replace({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'} ,regex = True)

print(data2['issue']) 
print(data2['d'])

#Concat the columns issue and d for creating a column for splitting the data into Train and test set

data2['period'] = data2['d'].map(str) + data2['issue'] 
data2['period'].unique()

#Sorting the data on the basis of period column

data2 = data2.sort_values('period') 
final_data = data2
del_column = ['str_split','issue','d','issue_d']
final_data = final_data.drop(labels = del_column, axis = 1)

# Provide the index to preiod for the split the data

final_data = final_data.set_index('period') 

#Creating the Training data set
Train_data = final_data.loc['200706':'201505',:] 
Train_data.index.unique()

#Creating the Training data set

Test_data = final_data.loc['201506':'201512',:]  
Test_data.index.unique()

#checking missing values in train and  test data

Train_data.isnull().sum()
Test_data.isnull().sum()

#Splitting the output variable into another dataframe

train_target = pd.DataFrame(Train_data['default_ind'])
train_target = train_target.astype(int)
test_target = pd.DataFrame(Test_data['default_ind'])
test_target = test_target.astype(int)

final_train = Train_data.iloc[:,0:66]
final_test = Test_data.iloc[:,0:66]
final_train.dtypes[final_train.dtypes != 'int64']

#Scaling the variables
from sklearn.preprocessing import StandardScaler

scaler_train = StandardScaler()
scaler_test = StandardScaler()

scaler_train.fit(final_train)
scaler_test.fit(final_test)

#Logisitc Regression --------------  98.85% ----------------------------------

from sklearn.linear_model import LogisticRegression

#Create a model

classifier=(LogisticRegression())

#Training the model
classifier.fit(final_train,train_target)

#predicting the values using logisitic Regression model
Y_pred=classifier.predict(final_test)

print(list(zip(test_target['default_ind'].values,Y_pred)))

#Comparing the results and checking the accuracy

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(test_target['default_ind'].values,Y_pred))
print(accuracy_score(test_target['default_ind'].values,Y_pred))
print(classification_report(test_target['default_ind'].values,Y_pred)) 

#Getting the ROC curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


y_pred_prob = classifier.predict_proba(final_test)[:,1]
fpr,tpr,threshold  = roc_curve(test_target,y_pred_prob)
plt.xlabel('Fpr')
plt.ylabel('Tpr')
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label = 'logistic regression')

roc_auc_score(test_target,y_pred_prob)

#Running Decision Tree Model------- 100 % -------------------------
#predicting using the DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()

model_DecisionTree.fit(final_train,train_target)

#fit the model on the data and predict the values
Y_dtree = model_DecisionTree.predict(final_test)

#print(Y_pred)

print(list(zip(test_target['default_ind'].values,Y_dtree)))

#confusion matrix

print(confusion_matrix(test_target['default_ind'].values,Y_dtree))

print(accuracy_score(test_target['default_ind'].values,Y_dtree))

print(classification_report(test_target['default_ind'].values,Y_dtree))

#running Random Forest Model--------------- 99.95 % ---------------------------------------

#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

#no of tress to be made
#no of tress to be made

model_RandomForest=RandomForestClassifier(20) 

#fit the model on the data and predict the values

model_RandomForest.fit(final_train,train_target)

Y_rfm=model_RandomForest.predict(final_test)

print(list(zip(test_target['default_ind'].values,Y_pred)))


print(confusion_matrix(test_target['default_ind'].values,Y_rfm))
print(accuracy_score(test_target['default_ind'].values,Y_rfm))
print(classification_report(test_target['default_ind'].values,Y_rfm)) 

#Knn algorithm--------------- 96.63 %--------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(final_train,train_target)
y_knn = knn.predict(final_test)                                 



print(confusion_matrix(test_target['default_ind'].values,y_knn))
print(accuracy_score(test_target['default_ind'].values,y_knn))
print(classification_report(test_target['default_ind'].values,y_knn))

#Plotting the ROC curve

knn_predict = knn.predict_proba(final_test)[:,1]
fpr,tpr,threshold = roc_curve(test_target,knn_predict)
plt.xlabel('Fpr')
plt.ylabel('Tpr')
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label = 'knn')

roc_auc_score(test_target,knn_predict)


#Ensemble Modelling-----------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#Creating the sub models

estimators=[]
model1=LogisticRegression()
estimators.append(('log',model1))

model2=DecisionTreeClassifier()
estimators.append(('cart',model2))

model3=SVC()
estimators.append(('svm',model3))


#Create the ensemble model
ensemble=VotingClassifier(estimators) #Voting Classifier refers that the method used for ensemble model is Voting and not Mean or Average weighted mean 

ensemble.fit(final_train,train_target)

Y_ensemble=ensemble.predict(final_test)

#Confusion Matrix
print(confusion_matrix(test_target['default_ind'].values,y_knn))
print(accuracy_score(test_target['default_ind'].values,y_knn))
print(classification_report(test_target['default_ind'].values,y_knn))






















































































































