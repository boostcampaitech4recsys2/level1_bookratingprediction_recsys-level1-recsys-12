# Book Rating Prediction

## π Project Overview
- λν λ΄μ© : Book Rating Prediction(μ¬μ©μκ° κ·Έλμ μ½μ μ±μ λΆμ¬ν νμ  λ°μ΄ν°λ₯Ό μ¬μ©ν΄μ μλ‘μ΄ μ±μ μΆμ²νμ λ μ΄λ μ λμ νμ μ λΆμ¬ν μ§ μμΈ‘νλ λ¬Έμ )
- νκ°μ§ν : RMSE (Root Mean Square Error)

## ποΈ Dataset
- users : 68,092λͺμ κ³ κ°(user)μ λν μ λ³΄λ₯Ό λ΄κ³  μλ λ©νλ°μ΄ν°
- books : 149,570κ°μ μ±(item)μ λν μ λ³΄λ₯Ό λ΄κ³  μλ λ©νλ°μ΄ν°
- train_ratings : 59,803λͺμ μ¬μ©μ(user)κ° 129,777κ°μ μ±(item)μ λν΄ λ¨κΈ΄ 306,795κ±΄μ νμ (rating) λ°μ΄ν°

## π Model
- CatBoost + LightGBM + Deep Learning model μμλΈ(7:1:2)

## π Result
- Public RMSE : 2.1286
- Private RMSE : 2.1241
- μ΅μ’ λ±μ : 3/14ν
![image](https://user-images.githubusercontent.com/64139953/200253536-1af45394-320a-4296-ab18-3e160d5e95ed.png)



