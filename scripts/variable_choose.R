library(mlbench)
library(caret)
library(recipes)


# pierwsze podejście
control <- trainControl(method="repeatedcv", number=10, repeats=3, verboseIter = TRUE)
model <- train(class~., 
               data=df.train, 
               method="rf",
               metric = "Accuracy", 
               preProcess="scale", trControl=control)

importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

# drugie podejście - to które finalnie wybraliśmy
control2 <- rfeControl(functions=rfFuncs, method="cv", number=10)

results <- rfe(df.train[,c(1, 2, seq(4,length(df.train), 1))], df.train[,3], sizes=c(1:49), rfeControl=control2)
predictors(results)
plot(results, type=c("g", "o"))

rfe_columns <- predictors(results)

write.csv(rfe_columns, "./data/output/rfe_columns.csv")

# PCA
recipe_obj <- recipe(class~ ., data = df.train) %>%
  step_scale(all_predictors()) %>%  # Standardize the predictors
  step_pca(all_predictors(), threshold = 0.90)  # Perform PCA with 90% variance threshold

recipe_obj <- prep(recipe_obj)

df_train_pca <- bake(recipe_obj, new_data = df.train)
df_test_pca <- bake(recipe_obj, new_data = df.test)

model

predictions <- predict(model, df_test_pca)