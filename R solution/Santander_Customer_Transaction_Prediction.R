

######################################### Remove all data from environment#########################################


rm(list=ls())



#############################################Setting Working Directory#############################################


setwd("C:/Users/Admin/Documents/R/R Scripts/Santander Project")




#################################################Loading Libraries#################################################


x = c("DMwR", "caret", "randomForest", "unbalanced", "C50", 'pROC','ROCR','cvAUC','e1071')
#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)



###############################################Importing the dataset###############################################


# Importing the dataset
train_data=read.csv('train.csv',header = T,na.strings = c(" ","",NA))
test_data=read.csv('test.csv',header = T,na.strings = c(" ","",NA))

#check the dimension of the dataset
dim(train_data)
dim(test_data)

#Dropping ID_code Column
train_data=subset(train_data,select=-ID_code)
print('Train & Test Data both contains 200000 rows, 200 feature columns [ var_0,var_1,...,var_199],one "ID_code" column. Train data has "target" variable as label ')


###############################################Data Pre Processing################################################


########## 1. Missing Value Analysis


sum(is.na(train_data))
sum(is.na(test_data))
print('No-Missing Values')

#No-Missing Values



numeric_data = subset(train_data,select=-target)
cnames = colnames(numeric_data)
cnames
rm(numeric_data)



########## 2. Outlier Analysis:- Replace all outliers with NA and then Imputing NA by Mean 


#Outlier Analysis
#Replace all outliers with NA 
for(i in cnames)
{
  val = train_data[,i][train_data[,i] %in% boxplot.stats(train_data[,i])$out]
  train_data[,i][train_data[,i] %in% val] = mean(train_data[i])
}
rm(val)





#Imputing NA by Mean 
for(i in cnames)
{
  train_data[,i][is.na(train_data[,i])] = mean(train_data[,i], na.rm = T)
}
sum(is.na(train_data))





#############################################Exploratory Data Analysis#############################################


#Target Class Distribution Bar Chart


barplot(table(train_data$target),main = 'Target Class Distribution',xlab = 'Target Class',ylab = 'Counts',col = c("orange", "green") ,cex.lab = 1.5)
legend("topright",c("Zero ( 0 )","One  ( 1 )"),bty="n", fill=c("orange", "green"))

##Target Class Imbalance Problem is present in the dataset as 10% data is accepted class & 90% data is rejected class

#Numerical Columns Distribution Histogram Chart

for (i in cnames)
{
  hist(train_data[,i],main=i)
}

##All the features in Dataset is pretty much normalised. 


#################################################Feature Selection#################################################

#Pearson Correlation Test


###### Lets assume Significance Level, SL is 0.05

###### Null Hypothesis:       Feature and Target doesnot have Linear Relationship : if calc_SL > assumed_SL
###### Alternate Hypothesis:  Feature and target have Linear Relationship : if calc_SL <= assumed_SL

column_toDrop=vector()
for (i in cnames)
{
  x=cor.test(train_data[,i], train_data$target,  method = "pearson")
  if (x$p.value >0.05)
  {
    column_toDrop=append(column_toDrop,i)
  }
}
rm(x)


#Dropping columns


###### Feature variable ['var_7', 'var_10', 'var_17', 'var_27', 'var_30', 'var_38', 'var_39', 'var_41', 'var_96', 'var_98', 'var_100', 'var_103', 'var_117','var_126', 'var_136', 'var_158', 'var_161', 'var_185']  shows zero correlation with the target variable. Thus dropping these feature.

train_data= train_data[, !colnames(train_data) %in% column_toDrop]
numeric_data = subset(train_data,select=-target)
cnames = colnames(numeric_data)
cnames
rm(numeric_data)




##################################################Feature Scaling##################################################

#Standardization

for(i in cnames){
  
  train_data[,i] = (train_data[,i] - mean(train_data[,i])) / sd(train_data[,i])
}




#################################################Train Test Split##################################################


set.seed(1234)
train.index = createDataPartition(train_data$target, p = .80, list = FALSE)
train = train_data[ train.index,]
test  = train_data[-train.index,]
rm(train.index)
rm(train_data)



####################### Error metric function


error_metric <- function(actual,predictions){
  
  #auc score
  auc_score=AUC(as.numeric(Predictions), test$target)
  
  precision=posPredValue(as.factor(Predictions), test$target, positive="1")
  recall=sensitivity(as.factor(Predictions), test$target, positive="1")
  
  f1_score=(2 * precision * recall) / (precision + recall)
  
  print('AUC Score:')
  print(auc_score)
  print('F1 Score:')
  print(f1_score)
  
  #(auc_score,f1_score)
}




#####################################################Modelling#####################################################

train$target = factor(train$target, levels = c(0, 1))
test$target = factor(test$target, levels = c(0, 1))



########## 1. Decision Tree

DT_model =C5.0(target ~., train, trials = 1, rules = TRUE)
Predictions = predict(DT_model, test[,-1], type = "class")

error_metric(test$target,Predictions)




########## 2. Logistic Regression


LR_model = glm(target ~ ., data = train, family = "binomial")
Predictions = predict(LR_model, newdata = test[,-1], type = "response")
Predictions = ifelse(Predictions > 0.5, 1, 0)

error_metric(test$target,Predictions)



########## 3. Random Forest

RF_model = randomForest(target ~ ., train, importance = TRUE, ntree = 10)
Predictions = predict(RF_model, test[,-1])
error_metric(test$target,Predictions)



########## 4. Naive Bayes

NB_model = naiveBayes(target ~ ., data = train)
Predictions = predict(NB_model, test[,-1], type = 'class')
error_metric(test$target,Predictions)

###### On the basis of Precision, Recall & ROC_AUC score, the best performing model is Naive Bayes model.

#AUC Score:  0.67, 
#F1 Score:  0.48, 


#The AUC & F1 score are quite average 

##### Since the data contains imbalanced target class, the above algorithms favors the negative class. 
##### Trying Sampling Techniques to improve the results.

#Sampling Techniques. [ Over Sampling, Under Sampling]



train_data = rbind(train,test)




########## 1. Under Sampling

########Creating Under sampled Data
x=train_data[train_data$target==1,]
y=train_data[train_data$target==0,]
y=y[sample(nrow(y), 20098),]
undersampled_data = rbind(x,y)



########Train Test Split

set.seed(1234)
train.index = createDataPartition(undersampled_data$target, p = .80, list = FALSE)
train = undersampled_data[ train.index,]
test  = undersampled_data[-train.index,]
rm(train.index)



########Naive Bayes Model

NB_model = naiveBayes(target ~ ., data = train)
Predictions = predict(NB_model, test[,-1], type = 'class')
error_metric(test$target,Predictions)



########## 2. Hybrid Sampling

########Creating Hybrid sampled data
x=train_data[train_data$target==1,]
y=train_data[train_data$target==0,]
hybridsampled_data=y[sample(nrow(y), 100000),]

for (i in 1:5)
{
  hybridsampled_data = rbind(hybridsampled_data,x)
}



########Train Test Split

set.seed(1234)
train.index = createDataPartition(hybridsampled_data$target, p = .80, list = FALSE)
train = hybridsampled_data[ train.index,]
test  = hybridsampled_data[-train.index,]
rm(train.index)



########Naive Bayes Model

NB_model = naiveBayes(target ~ ., data = train)
Predictions = predict(NB_model, test[,-1], type = 'class')
error_metric(test$target,Predictions)



#Hybrid Sampling performs better than Under Sampling. Thus choosing Naive Bayes Model Developed with Hybrid Sampled Data

#Naive Bayes Result:
#F1 Score:  0.81
#AUC Score: 0.81


######Saving the Final Model

# save the model to disk
saveRDS(NB_model, "./final_model.rds")


######Saving the Column to Drop List 

# save the model to disk
saveRDS(column_toDrop, "./column_toDrop_list.rds")

