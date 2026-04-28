# Уравнения Навье-Стокса (2D, несжимаемая жидкость)

## Уравнение неразрывности
```
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$
```
## Уравнение количества движения по оси x

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = 
-\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

## Уравнение количества движения по оси y

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = 
-\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
