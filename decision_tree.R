str(iris)

set.seed(1234)
ind = sample(2, nrow(iris), replace=TRUE, prob=c(0.7,0.3))
traindata = iris[ind==1,]
testdata = iris[ind==2,]

library(party)
myFormula = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_ctree = ctree(myFormula, data = traindata)
table(predict(iris_ctree),traindata$Species)

print(iris_ctree)

plot(iris_ctree)

data("bodyfat", package="TH.data")
dim(bodyfat)
attributes(bodyfat)
str(bodyfat)

set.seed(1234)
ind = sample(2, nrow(bodyfat), replace=TRUE, prob=c(0.7, 0.3))
bodyfat.train = bodyfat[ind==1,]
bodyfat.test = bodyfat[ind==2,]
library(rpart)
myFormula = DEXfat ~ age + waistcirc + hipcirc + elbowbreadth + kneebreadth
bodyfat_rpart = rpart(myFormula, data = bodyfat.train, control = rpart.control(minsplit = 10))
attributes(bodyfat_rpart)
print(bodyfat_rpart$cptable)
print(bodyfat_rpart)
plot(bodyfat_rpart)
text(bodyfat_rpart, use.n=T)
opt = which.min(bodyfat_rpart$cptable[,"xerror"])
cp = bodyfat_rpart$cptable[opt,"CP"]
bodyfat_prune = prune(bodyfat_rpart, cp = cp)
print(bodyfat_prune)
plot(bodyfat_prune)
text(bodyfat_prune, use.n=T)
DEXfat_pred = predict(bodyfat_prune, newdata=bodyfat.test)
xlim = range(bodyfat$DEXfat)
plot(DEXfat_pred ~ DEXfat, data=bodyfat.test, xlab="Observed", ylab="Predicted", ylim=xlim, xlim=xlim)
abline(a=0, b=1)
