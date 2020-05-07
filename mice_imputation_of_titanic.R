setwd("C:/Users/erikj/Dropbox/TitanicData")

library(dplyr)
library(mice)
library(gbm)


# load and format training data
train = read.csv("train.csv")
#View(train)
train_PassengerId = train[, 1]
train_Survived = as.factor(train[, 2])
train = train[, c(1,3,5,6,7,8,10,12)]

str(train)

train = train %>% mutate(
  Pclass = as.factor(Pclass),
  Sex = as.factor(Sex),
  Age = as.numeric(Age),
  SibSp = as.factor(SibSp),
  Parch = as.factor(Parch),
  Fare = as.numeric(Fare),
  Embarked = as.factor(Embarked)
)

str(train)

# load and format testing data
test = read.csv("test.csv")
test_PassengerId = test[, 1]
test = test[, c(1,2,4,5,6,7,9,11)]
test = test %>% mutate(
  Pclass = as.factor(Pclass),
  Sex = as.factor(Sex),
  Age = as.numeric(Age),
  SibSp = as.factor(SibSp),
  Parch = as.factor(Parch),
  Fare = as.numeric(Fare),
  Embarked = as.factor(Embarked)
)

train$Embarked[which(train$Embarked=="")] = NA

str(test)

# impute ONLY TRAINING data with mice
init = mice(train, maxit=0)
meth = init$method
predM = init$predictorMatrix

predM[, c("PassengerId")] = 0 # remove this as a predictor during imputation
meth[c("PassengerId")] = "" # do not impute this variable

meth[c("Age", "Fare")] = "norm" # numeric
meth[c("Sex")] = "logreg" # factor with 2 levels
meth[c("Pclass", "SibSp", "Parch", "Embarked")] = "polyreg" # factor with more than 2 levels

set.seed(112)
imputed = mice(train, method=meth, predictorMatrix=predM, m=5)
train_imputed = complete(imputed)
sapply(train_imputed, function(x) sum(is.na(x)))

# now we combine train_imputed and test to impute the complete dataset
# then we separate out the imputed_test
complete_data = rbind(train_imputed, test)
str(train_imputed)

init2 = mice(complete_data, maxit=0)
meth2 = init2$method
predM2 = init$predictorMatrix

predM2[, c("PassengerId")] = 0
meth2[c("PassengerId")] = ""

meth2[c("Age", "Fare")] = "norm" # numeric
meth2[c("Sex")] = "logreg" # factor with 2 levels
meth2[c("Pclass", "SibSp", "Parch", "Embarked")] = "polyreg" # factor with more than 2 levels

imput = mice(complete_data, method=meth2, predictorMatrix=predM2, m=5)
complete_imputed = complete(imput)
sapply(complete_imputed, function(x) sum(is.na(x)))

test_imputed = complete_imputed[test_PassengerId, ] # test data with imputed values

# create a generic boost model
PassengerId = train_PassengerId
Survived = train_Survived

train_imputed = train_imputed[,-1]
train_imputed = cbind(Survived, train_imputed)

train_imputed$Survived = as.character(train_imputed$Survived)

train_boost = gbm(formula=Survived~., distribution="bernoulli", data=train_imputed, n.trees=5000, interaction.depth=4, shrinkage=0.1)
pred_boost = predict(train_boost, newdata=test_imputed, n.trees=5000, type="response")
pred_boost = ifelse(pred_boost > 0.5, 1, 0)
df = data.frame(PassengerId=test_PassengerId, Survived=pred_boost)
write.csv(df, file="correct_imputed.csv", col.names=FALSE, row.names=FALSE)

# this did not work as it produced a 69% accuracy result, worse than the previous mice imputation and NA == mean()

