library(dplyr)

rm(list=ls())

df <- read.csv("data/input/c2.csv")

df <- df %>%
  mutate_if(is.character, function(x) gsub("'", "", x))
colSums(is.na(df))

rm(list=ls())

head(df)

# Write original order of columns for further purposes
original_column_order <- colnames(df)

# Missing values

## Getting the number of missing values in each column
num_missing = colSums(is.na(df))

# Excluding columns that contains 0 missing values
num_missing = num_missing[num_missing > 0]
num_missing


training_obs <- createDataPartition(df$id, 
                                    p = 0.7, 
                                    list = FALSE) 
df.train <- df[training_obs,]
df.test  <- df[-training_obs,]

install.packages("caret")
rm(list=ls())
install.packages("plotfunctions")


 df %>%
  group_by(job, class) %>%
  summarise(count = n())

ggplot(data = hist, aes(x = job, y = count, fill = class)) +
  geom_col(position = position_dodge())




