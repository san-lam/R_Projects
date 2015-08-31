data(iris)
head(iris)
iris.features = iris
iris.features$Species = NULL
results = kmeans(iris.features, 3)
table(iris$Species, results$cluster)
plot(iris[c("Petal.Length", "Petal.Width")], col = results$cluster)
plot(iris[c("Petal.Length", "Petal.Width")], col = iris$Species)
plot(iris[c("Sepal.Length", "Sepal.Width")], col = results$cluster)
plot(iris[c("Sepal.Length", "Sepal.Width")], col = iris$Species)
