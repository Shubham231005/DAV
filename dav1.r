#User-defind dataset
X <- c(1,2,3,4,5)
y <- c(12,19,29,37,45)

#Create data frame
data <- data.frame(X,y)

#Create linear regression model
model <- lm(y~X,data=data)

#Calculate slope and intercept
intercept <- coef(model)[1] 
slope <- coef(model)[2]

#Predict for X =7
new_data <- data.frame(X=6)
y_new <- predict(model,new_data)

#Print results
cat("Slope(m):",slope,"\n")
cat("Intercept(b)",intercept,"\n")
cat("Predicted value when X=6:",y_new,"\n")

#Plot with extended limits
plot(X,y,
     main = "Simple Linear Regression",
     Xlab = "Years",
     ylab = "Expenditure",
     pch = 20,
     xlim = c(1,10),
     ylim = c(10,70)
  
)

#Regression line
abline(model, col ="blue")

#Predicted point(red dot)
points(6,y_new,pch =16,col="red")

text(3,78,labels = paste("y=",round(slope,2),"X+",round(intercept,2)),col = "darkgreen")
