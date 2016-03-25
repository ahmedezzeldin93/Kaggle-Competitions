import pandas as pd 
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier


titanic_train_df = pd.read_csv('train.csv')
titanic_test_df = pd.read_csv('test.csv')

titanic_train_df=titanic_train_df.drop(['PassengerId','Name','Ticket'],axis=1)
titanic_test_df=titanic_test_df.drop(['Name','Ticket'],axis=1)

#Pclass Column Manipulation
pclass_dummies_train=pd.get_dummies(titanic_train_df.Pclass)
pclass_dummies_train.columns=['Class1','Class2','Class3']
pclass_dummies_test=pd.get_dummies(titanic_test_df.Pclass)
pclass_dummies_test.columns=['Class1','Class2','Class3']
titanic_train_df.drop(['Pclass'],axis=1,inplace=True)
titanic_test_df.drop(['Pclass'],axis=1,inplace=True)
titanic_train_df=titanic_train_df.join(pclass_dummies_train)
titanic_test_df=titanic_test_df.join(pclass_dummies_test)

#Sex Column Manipulation
def male_female_child(passenger):
    age,sex = passenger
    if age < 16:
        return 'Child'
    else:
        return sex
titanic_train_df['Person']=titanic_train_df[['Age','Sex']].apply(male_female_child,axis=1)
titanic_test_df['Person']=titanic_test_df[['Age','Sex']].apply(male_female_child,axis=1)
sex_dummies_train=pd.get_dummies(titanic_train_df.Person)
sex_dummies_train.columns=['Male','Female','Child']
sex_dummies_test=pd.get_dummies(titanic_test_df.Person)
sex_dummies_test.columns=['Male','Female','Child']
titanic_train_df.drop(['Sex','Person'],axis=1,inplace=True)
titanic_test_df.drop(['Sex','Person'],axis=1,inplace=True)
titanic_train_df=titanic_train_df.join(sex_dummies_train)
titanic_test_df=titanic_test_df.join(sex_dummies_test)
titanic_train_df.drop(['Male'],axis=1,inplace=True)
titanic_test_df.drop(['Male'],axis=1,inplace=True)

#Family Column Manipulation
titanic_train_df['Family']=titanic_train_df.Parch+titanic_train_df.SibSp
titanic_test_df['Family']=titanic_test_df.Parch+titanic_test_df.SibSp
titanic_train_df.Family.loc[titanic_train_df.Family>0]=1
titanic_train_df.Family.loc[titanic_train_df.Family==0]=0
titanic_test_df.Family.loc[titanic_test_df.Family>0]=1
titanic_test_df.Family.loc[titanic_test_df.Family==0]=0
titanic_train_df.drop(['SibSp','Parch'],axis=1,inplace=True)
titanic_test_df.drop(['SibSp','Parch'],axis=1,inplace=True)

# Embarked Column Manipulation
titanic_train_df.Embarked.map({'S':0,'C':1,'Q':3})
titanic_train_df['Embarked']=titanic_train_df.Embarked.fillna('S')
embarked_dummies=pd.get_dummies(titanic_train_df.Embarked)
titanic_train_df=titanic_train_df.join(embarked_dummies)
titanic_train_df.drop('Embarked',inplace=True,axis=1)
embarked_dummies_test=pd.get_dummies(titanic_test_df.Embarked)
titanic_test_df=titanic_test_df.join(embarked_dummies_test)
titanic_test_df.drop('Embarked',inplace=True,axis=1)


#Age Column Manipulation
train_age_mean=titanic_train_df.Age.mean()
train_age_std=titanic_train_df.Age.std()
train_age_count=titanic_train_df.Age.isnull().sum()
test_age_mean=titanic_test_df.Age.mean()
test_age_std=titanic_test_df.Age.std()
test_age_count=titanic_test_df.Age.isnull().sum()
rand_1=np.random.randint(train_age_mean-train_age_std,train_age_mean+train_age_std,size=train_age_count)
rand_2=np.random.randint(test_age_mean-test_age_std,test_age_mean+test_age_std,size=test_age_count)
titanic_train_df["Age"][np.isnan(titanic_train_df["Age"])] = rand_1
titanic_test_df["Age"][np.isnan(titanic_test_df["Age"])] = rand_2
titanic_test_df.Age[titanic_train_df.Age.isnull()]


#Fare Column Manipulation
titanic_test_df.Fare.fillna(titanic_test_df.Fare.median(),inplace=True)

#Cabin Column Manipulation
titanic_train_df.drop('Cabin',inplace=True,axis=1)
titanic_test_df.drop('Cabin',inplace=True,axis=1)

X_train=titanic_train_df.drop('Survived',axis=1)
Y_train=titanic_train_df.Survived
X_test=titanic_test_df.drop('PassengerId',axis=1)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
score = random_forest.score(X_train, Y_train)
print("Random forest Score: ",score)

submission=pd.DataFrame({"PassengerId":titanic_test_df.PassengerId,"Survived":Y_pred})
submission.to_csv('titanic.csv',index=False)