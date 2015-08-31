# Assign the training set
train <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"))
  
# Assign the testing set
test <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"))
  
# Make sure to have a look at your training and testing set
print(train)
print(test)

# Your train and test set are still loaded in
str(train)
str(test)

# Passengers that survived vs passengers that passed away
table(train$Survived)
prop.table(table(train$Survived))

# Males & females that survived vs males & females that passed away
table(train$Sex, train$Survived)
prop.table(table(train$Sex, train$Survived),1)

# Create the column child, and indicate whether child or no child
train$Child = NA
train$Child[train$Age < 18] = 1
train$Child[train$Age >= 18] = 0

# Two-way comparison
table(train$Child, train$Survived)
prop.table(table(train$Child, train$Survived), 1)

# prediction based on gender 
test_one <- test
test_one$Survived = NA
test_one$Survived[test_one$Sex == 'male'] = 0
test_one$Survived[test_one$Sex == 'female'] = 1

##################################################################

# Load in the R package  
library(rpart)

# Build the decision tree
my_tree_two <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")

# Visualize the decision tree using plot() and text()
plot(my_tree_two)
text(my_tree_two)

# Load in the packages to create a fancified version of your tree
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Time to plot your fancified tree
fancyRpartPlot(my_tree_two)

# Make your prediction using the test set
my_prediction <- predict(my_tree_two, newdata=test, type="class")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Check that your data frame has 418 entries
nrow(my_solution)

# Write your solution to a csv file with the name my_solution.csv
write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)

# Create a new decision tree my_tree_three
my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class", control = rpart.control(minsplit = 50, cp = 0))
  
# Visualize your new decision tree
fancyRpartPlot(my_tree_three)

# create a new train set with the new variable
train_two <- train
train_two$family_size <- train_two$SibSp + train_two$Parch + 1

# Create a new decision tree my_tree_three
my_tree_four <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size, data=train_two, method="class")
  
# Visualize your new decision tree
fancyRpartPlot(my_tree_four)

# train_new and test_new are available in the workspace
str(train_new)
str(test_new)

# Create a new model `my_tree_five`
my_tree_five <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data=train_new, method="class")

# Visualize your new decision tree
fancyRpartPlot(my_tree_five)

# Make your prediction using `my_tree_five` and `test_new`
my_prediction <- predict(my_tree_five, newdata=test_new, type="class")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file="my_solution.csv", row.names = FALSE)

#####----------------Random Forest------------------------######
# All data, both training and test set
all_data

# Passenger on row 62 and 830 do not have a value for embarkment. 
# Since many passengers embarked at Southampton, we give them the value S.
# We code all embarkment codes as factors.
all_data$Embarked[c(62,830)] = "S"
all_data$Embarked <- factor(all_data$Embarked)

# Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
all_data$Fare[1044] <- median(all_data$Fare, na.rm=TRUE)

# How to fill in missing Age values?
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# This time you give method="anova" since you are predicting a continuous variable.
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size, data=all_data[!is.na(all_data$Age),], method="anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, all_data[is.na(all_data$Age),])

# Split the data back into a train set and a test set
train <- all_data[1:891,]
test <- all_data[892:1309,]


# train and test are available in the workspace
str(train)
str(test)

# Load in the package
library(randomForest)

# Train set and test set
str(train)
str(test)

# Set seed for reproducibility
set.seed(111)

# Apply the Random Forest Algorithm
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data = train, ntree = 1000, importance = TRUE)

# Make your prediction using the test set
my_prediction <- predict(my_forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId=test$PassengerId, Survived=my_prediction)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file="my_solution.csv", row.names = FALSE)

