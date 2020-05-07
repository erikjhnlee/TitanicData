setwd("C:/Users/erikj/Dropbox/TitanicData")

# this time we are goingto use the kNN (k-nearest neightbors) imputation from the VIM package
# since there is no predict method for kNN we may again combine the training and test datasets to imput the test data

library(VIM)
library(randomForest)
library(dplyr)

# import our cleaned training and test data
train = read.csv("train_clean.csv")
test = read.csv("test_clean.csv")

# set aside the pid for later. just keep relavent columns for imputation and classification
train_pid = train$PassengerId
train = train[, -1]

test_pid = test$PassengerId
test = test[, -1]

# check the class types for each column
str(train)
str(test)

# we are going to mutate each column. categorical variables will be factors
train = train %>% mutate(
  Survived = as.factor(Survived),
  Pclass = as.factor(Pclass),
  SibSp = as.factor(SibSp),
  Parch = as.factor(Parch)
)

test = test %>% mutate(
  Pclass = as.factor(Pclass),
  SibSp = as.factor(SibSp),
  Parch = as.factor(Parch)
)

# check for nulls in training and test datasets
sapply(train, function(x)sum(is.na(x)))
sapply(test, function(x)sum(is.na(x)))

# we see that for training Age is missing 177 values and Embarked is missing 2 values (20% of data has NAs)
# for test Age is missing 86 values and Fare is missing 1 value (20% of data has NAs)
# now we apply kNN() from VIM to the training data. filling in the nulls for 

train_knn = kNN(train, variable=c("Age", "Embarked"), k=3)
train_knn = subset(train_knn, select=Survived:Embarked)

# in the previous imputations with mice, the training set was used with the test set to impute the complete data
# however, as pointed out by a StackOverflow thread, imputing the training and test data sets will result
# in overfitting of the model. instead, we would like to impute each set separately as a past/future practice,
# where the training data was aquired in the past and the test data is aquired in the future separately and unlinked.
test_knn = kNN(test, variable=c("Age", "Fare"))
test_knn = subset(test_knn, select=Pclass:Embarked)

# now that we have imputed our data, we will fit an unpruned decision tree on the training data and then get results
# for the test data. then, we will compare these results to the mean and mice imputations
suppressWarnings(library(tree))
tree_train = tree(formula=Survived~., data=train_knn)
pred_tree = predict(tree_train, newdata=train_knn, type="class")
t = table(pred_tree, train_knn$Survived)
(t[1,1] + t[2,2])/sum(t) # this error rate on the training data is very similar to the mice imputation and tree model

# we can use cross validation from cv.tree() to check which pruned size produces the lowest deviance
# in this case, deviance is the classification error rate which is 1-max_k(E),
# where the error is the number percentage of observations that do not belong to the maximum class K
tree_cv = cv.tree(tree_train, FUN=prune.misclass)
tree_cv

# it appears that size 9 and 10 trees have a tie with lowest deviance. so we will go with the unpruned full tree
pred_test = predict(tree_train, newdata=test_knn, type="class")
pred_test = as.numeric(as.character(pred_test))
test_df = data.frame(PassengerId=test_pid, Survived=pred_test)
write.csv(x=test_df, file="tree_knn.csv", row.names=FALSE)

# the imputation and model above returned an accuracy of 0.77511 or 77.511% accuracy
# next, we would want to see if the k-value for kNN imputation has an affect on our accuracy
# let us try to test k values of 1-8 and use cross-validation 
# we will make this a function to make this easier to test values again

cv_knn = function(data_raw=NULL, k_val, folds=10){
  impute_cols = names(which(sapply(data_raw, function(x)sum(is.na(x)))>0))
  keep_cols = colnames(data_raw)
  train = kNN(data_raw, variable=impute_cols, k=k_val)
  train = subset(train, select=keep_cols)
  
  avg_acc = numeric()
  for(f in 1:folds){
    rows = rep(1:folds, times=nrow(train))
    training = train[rows==f, ]
    testing = train[rows!=f, ]
    form = paste(colnames(training)[1], "~.", sep="")
    tree_model = tree(formula=form, data=training)
    pred = predict(tree_model, newdata=testing, type="class")
    tbl = table(pred, testing[, 1])
    acc = (tbl[1,1]+tbl[2,2])/sum(tbl)
    avg_acc = c(avg_acc, acc)
  }
  return(mean(avg_acc))
}

res = numeric()
for(i in 1:10){
  vals = numeric()
  for(j in 1:20){
    v = cv_knn(data_raw=train, k_val=i, folds=0)
    vals = c(vals, v)
  }
  res = c(res, mean(vals))
}

plot(x=1:10, y=res)
best_k = which.max(res) 
# we see from running 20 iterations for k from 1 to 10 that k=1 and k=2 have the highest level of accuracy
# we may consider using k=1 for the imputation and then fitting a tree
imp_vars = names(which(sapply(train, function(x)sum(is.na(x)) > 0)))
train_k1 = kNN(train, variable=imp_vars, k=best_k)
train_k1 = train_k1[, 1:ncol(train)]

imp_vars_2 = names(which(sapply(test, function(x)sum(is.na(x)) > 0)))
test_k1 = kNN(test, variable=imp_vars_2, k=best_k)
test_k1 = test_k1[, 1:ncol(test)]

tree_k1 = tree(formula=Survived~., data=train_k1)
pred_k1 = predict(tree_k1, newdata=test_k1, type="class")
pred_k1 = as.numeric(as.character(pred_k1))
df_k1 = data.frame(PassengerId=test_pid, Survived=pred_k1)
write.csv(df_k1, "k1_res.csv", row.names=FALSE)

# very exciting that we recieved a score accuracy of 0.79904 or 79.904%. this is extremely close to our desired score
# of 80% or higher. this gives the advantage for imputation methods to kNN over mean and mice imputations
# in the next file, we will use different models such as bagging, RandomForest and Ridge Regression
# to compare which produces best results with kNN
