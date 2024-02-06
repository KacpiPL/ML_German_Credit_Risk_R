library(mlbench)
library(caret)
library(recipes)

rm(list=ls())

df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

# rfe requires factor as target variable
df.train[,1] <- as.factor(df.train[,1])

# Feature selection using Recursive Feature Elimination (RFE)

# Define seeds for rfeControl
# If we use doParallel packae and want to make our code reproducible we should pass seeds argument to rfeControl function
# It requires a list where each element corresponds to a specific resampling iteration's seed, 
# and the last element is a seed for the overall process.
# Assuming 10-fold CV, we need a list of 11 elements: 10 for each fold and 1 for the overall process
set.seed(123456789)
numFolds <- 10
seedsList <- lapply(1:numFolds, function(x) sample.int(1000, 49)) # Generate random seeds for each fold

set.seed(123456789)
seedsList[[length(seedsList) + 1]] <- sample.int(1000, 1) # Add one more seed for the overall process

## Define rfeControl object
control <- rfeControl(functions=rfFuncs, method="cv", number=10, seeds = seedsList)

# # Perform feature selection with rfe
# Target variable "class" is on the first position, so we set the parameters accordingly
set.seed(123456789)
results <- rfe(df.train[, seq(2, length(df.train), 1)],             # predictor variables 
               df.train[, 1],                                       # target variable for prediction
               sizes=c(1:49),                                       # the subset sizes to evaluate
               rfeControl=control)

predictors(results)
plot(results, type=c("g", "o"))

rfe_columns <- predictors(results)
write.csv(rfe_columns, "./data/output/rfe_columns.csv")
