# load files and make dummy file for train data
train = read.csv("train.csv")
test = read.csv("test.csv")

train_dummy = data.frame(train)

# first to cleanr the data we need to see if any of the observations are missing values for variables
nulls = c()
for(col in 1:ncol(train)){
  nulls = c(nulls, sum(is.na(train[,col])))
}

vars = colnames(train)
df.nulls = data.frame(Variables=vars, NullCount=nulls)

# according to the dataframe corresponding to the null count, Age has 177 nulls
# let's train on a complete dataset by removing nulls in Age
# we can always train on the full training data if we do not find age as significant
train_dummy = train_dummy[!is.na(train_dummy$Age),]
