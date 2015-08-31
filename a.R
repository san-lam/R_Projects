ss1=sum(fit1$residuals^2)
ss2=sum(fit2$residuals^2)
df1=209
df2=206

((ss1-ss2)/(df1-df2))/(ss2/df2)

1-pf(38.1243,df1-df2,df2)

