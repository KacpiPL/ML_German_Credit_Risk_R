getAccuracyAndGini <- function(model, 
                               data, 
                               target_variable = "UCURNINS",
                               predicted_class = "Yes") {
  
  # funkcja dla modelu zapisanego jako wynik 
  # funkcji train() obliczy 
  # accuracy, specificity, sensitivity i Gini = 2 * AUC - 1
  
  require(pROC)
  data <- data %>% as.data.frame()
  
  # wygeneruj prawdopodobieństwa poziomu "predicted_class"
  forecasts_p <- predict(model, data,
                         type = "prob")[, predicted_class]
  
  # i samą przewidywaną kategorię
  if (any(class(model) == "train")) {
    forecasts_c <- predict(model, data) 
  } else forecasts_c <- predict(model, data, type = "class")
  
  # wartości rzeczywiste - pull() zamienia obiekt tibble w wektor
  real <- data[, target_variable]
  
  # pole pod wykresem ROC
  AUC <- roc(predictor = forecasts_p,
             response = (real == predicted_class),
             quiet = T)
  
  # tabela klasyfikacji i miary na niej oparte
  table <- confusionMatrix(forecasts_c,
                           real,
                           predicted_class) 
  # zbieramy w ostateczny wynik
  result <- c(table$overall[1], # cccuracy
              table$byClass[1:2], # sens, spec
              Gini = 2 * AUC$auc - 1)
  
  return(result)
  
}
