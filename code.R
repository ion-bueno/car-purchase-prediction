# Fix the seed to ensure reproducibility
set.seed(1)


## ------- Import the dataset and prepare it to work -------

cars =  read.csv("dataset.csv")

vars <- c("Gender", "Age", "Annual.Salary", "Credit.Card.Debt",
          "Net.Worth", "Car.Purchase.Amount")

cars <- subset(cars, select = vars)

colnames(cars)[1] <- "gender"
colnames(cars)[2] <- "age"
colnames(cars)[3] <- "salary"
colnames(cars)[4] <- "card_debt"
colnames(cars)[5] <- "net_worth"
colnames(cars)[6] <- "purchase_amount"



## ------- Divide the dataset into training and test samples -------

n <- nrow(cars)
n_train <- floor(.96*n)
n_test <- n - n_train

# Obtain the training and test partitions
i_train <- sort(sample(1:n,n_train))
i_test <- setdiff(1:n,i_train)

cars_test <- cars[i_test,]
cars <- cars[i_train,]

attach(cars)



## ------- Data visualization -------

library(ggplot2)
library(GGally)

my_fn <- function(data, mapping, ...){
  p <- ggplot(data = data, mapping = mapping) + 
    geom_point() + 
    geom_smooth(method=loess, fill="red", color="red", ...) +
    geom_smooth(method=lm, fill="blue", color="blue", ...)
  p
}

ggpairs(cars,columns = 1:6, upper = list(continuous = my_fn), lower = list(continuous = my_fn))



## ------- Simple linear regression models -------

modAge <- lm(purchase_amount ~ age, data = cars)
summary(modAge)

modSalary <- lm(purchase_amount ~ salary, data = cars)
summary(modSalary)



## ------- Linear models -------

# Including more predictors
modAll <- lm(purchase_amount ~ ., data = cars)
summary(modAll)

modAllSig <- lm(purchase_amount ~ . -card_debt -gender, data = cars)
summary(modAllSig)

# Relations study
corrplot::corrplot(cor(cars), addCoef.col = "grey")

car::vif(modAll)

# Gender simple model
modGender <- lm(purchase_amount ~ gender, data = cars)
summary(modGender)

# Logistic regression with gender: computed with glm and family = "binomial" 
plot(purchase_amount, gender)

modLogGender <- glm(gender ~ purchase_amount, family = "binomial", data = cars)

x <- seq(min(purchase_amount), max(purchase_amount), l = 1000)
y <- exp(-(modLogGender$coefficients[1] + modLogGender$coefficients[2] * x))
y <- 1 / (1 + y)
lines(x, y, col = 2, lwd = 2) # Draw the fitted logistic curve



## ------- ANOVA decomposition -------

simpleAnova <- function(object, ...) {
  # This function computes the simplified anova from a linear model
  # Compute anova table
  tab <- anova(object, ...)
  # Obtain number of predictors
  p <- nrow(tab) - 1
  # Add predictors row
  predictorsRow <- colSums(tab[1:p, 1:2])
  predictorsRow <- c(predictorsRow, predictorsRow[2] / predictorsRow[1])
  # F-quantities
  Fval <- predictorsRow[3] / tab[p + 1, 3]
  pval <- pf(Fval, df1 = p, df2 = tab$Df[p + 1], lower.tail = FALSE)
  predictorsRow <- c(predictorsRow, Fval, pval)
  # Simplified table
  tab <- rbind(predictorsRow, tab[p + 1, ])
  row.names(tab)[1] <- "Predictors"
  return(tab)
}

simpleAnova(modAll)
simpleAnova(modAllSig)

# Stepwise model selection with BIC
modAllBIC <- MASS::stepAIC(modAll, direction = "both", k = log(nrow(cars)))
summary(modAllBIC)



## ------- Model with interations of predictors -------

modInt <- MASS::stepAIC(object = lm(purchase_amount ~ ., data = cars),
              scope = purchase_amount ~ .^2, k = log(nrow(cars)), trace = 1)
summary(modInt)

# Interactions with gender
ggpairs(cars, columns = c(1:5), aes(color = gender),
        upper = list(continuous = "points"))


modIntGender1 <- lm(purchase_amount ~  .* gender, data = cars)
summary(modIntGender1)

# The components given by the interactions are not significant

modIntGender2 <- lm(purchase_amount ~  .: gender, data = cars)
summary(modIntGender2)

# The R^2 decreases significantly if only interactions are used

fit1 <- lm(purchase_amount ~ .^2, data = cars)
fit2 <- lm(purchase_amount ~ ., data = cars)
MASS::stepAIC(fit2,direction="both",scope=list(upper=fit1,lower=fit2), k = log(nrow(cars)))

# The interactions are removed



## ------- Ridge regression -------

# we separate the target variable from the predictors
X <- cars[-c(6)]
y <- purchase_amount
library(glmnet)
ridgeMod <- glmnet(x = X, y = y, alpha = 0)
summary(ridgeMod)
plot(ridgeMod, label = TRUE, xvar = "lambda")

# we generate the ridge regression tuning the parameteres using cross-validation
# with 10 folds
kcvRidge <- cv.glmnet(x = as.matrix(X), y = y, alpha = 0, nfolds = 10)

# The lambda that minimizes the CV error is
kcvRidge$lambda.min

# The minimum CV error
kcvRidge$cvm[indMin]

# Minimum occurs at one extreme of the lambda grid in which CV is done
range(kcvRidge$lambda)
# The grid was automatically selected, but can be manually inputted
lambdaGrid <- 10^seq(log10(kcvRidge$lambda[1]), log10(0.01),
                     length.out = 150) # log-spaced grid
kcvRidge2 <- cv.glmnet(x = as.matrix(X), y = y, nfolds = 10, alpha = 0,
                       lambda = lambdaGrid)
# New plot
plot(kcvRidge2)

# The lambda that minimizes the CV error is
kcvRidge2$lambda.min
# Range lambdas
range(kcvRidge2$lambda)



## ------- Lasso regression -------

# Call to the main function using alpha = 1 for lasso regression
lassoMod <- glmnet(x = X, y = y, alpha = 1)
# Same defaults as before, same object structure

# Plot of the solution path -- now the paths are not smooth when decreasing to
# zero (they are zero exactly). This is a consequence of the l1 norm
plot(lassoMod, xvar = "lambda", label = TRUE)

kcvLasso <- cv.glmnet(x = as.matrix(X), y = y, alpha = 1, nfolds = 10)

# The lambda that minimizes the CV error is
kcvLasso$lambda.min

# The "one standard error rule" for lambda
kcvLasso$lambda.1se

# Location of both optimal lambdas in the CV loss function
indMin <- which.min(kcvLasso$cvm)
plot(kcvLasso)
abline(h = kcvLasso$cvm[indMin] + c(0, kcvLasso$cvsd[indMin]))

lambdaGrid <- 10^seq(log10(kcvLasso$lambda[1]), log10(0.1),
                     length.out = 150) # log-spaced grid
ncvLasso <- cv.glmnet(x = as.matrix(X), y = y, alpha = 1, nfolds = nrow(X),
                      lambda = lambdaGrid)

# Location of both optimal lambdas in the CV loss function
plot(ncvLasso)

kcvLasso$lambda.min
range(kcvLasso$lambda)

# In the case of Lasso we have the same problem as the one presented in the 
# Ridge regression, that's why we generate a lambda grid in order to look
# for the optimal lambda
lambdaGrid <- 10^seq(log10(kcvLasso$lambda[1]), log10(0.001),
                     length.out = 150) # log-spaced grid
kcvLasso2 <- cv.glmnet(x = as.matrix(X), y = y, nfolds = 10, alpha = 1,
                       lambda = lambdaGrid)
plot(kcvLasso2)
kcvLasso2$lambda.min

indMin2 <- which.min(kcvLasso2$cvm)
abline(h = kcvLasso2$cvm[indMin2] + c(0, kcvLasso2$cvsd[indMin2]))

range(kcvLasso2$lambda)

# with the cross-validation object we get the model in order to get the BIC
modLasso = kcvLasso2$glmnet.fit
modRidge = kcvRidge2$glmnet.fit


plot(modLasso, label=TRUE, xvar="lambda")
plot(modRidge, label=TRUE, xvar="lambda")


predsLasso = predict(modLasso, type="coefficients", 
        s=c(kcvLasso2$lambda.min, kcvLasso2$lambda.1se))[-1, ] != 0

predsRidge = predict(modRidge, type="coefficients", 
                     s=c(kcvRidge2$lambda.min, kcvRidge2$lambda.1se))[-1, ] != 0

# we generate two different models in each case, one for each value of lambda
# (lambda.min and lambda.1se)
x11 <- X[, predsLasso[, 1]]
x12 <- X[, predsLasso[, 2]]
x21 <- X[, predsRidge[, 1]]
x22 <- X[, predsRidge[, 2]]

# Least squares fit with variables selected by lasso
modLasso1 <- lm(y ~ as.matrix(x11))
modLasso2 <- lm(y ~ as.matrix(x12))

modRidge1 <- lm(y ~ as.matrix(x21))
modRidge2 <- lm(y ~ as.matrix(x22))

# We get a similar value of BIC for all the models
BIC(modAll, modAllSig, modLasso1, modLasso2, modRidge1, modRidge2)



## ------- Prediction on test partition -------

# Predictions using the best model
predictions = predict(modAllSig, newdata = cars_test, interval = "prediction")

# True label
y_true = cars_test$purchase_amount

preds_vs_true = cbind(predictions, y_true)
preds_vs_true
