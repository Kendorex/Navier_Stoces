# Уравнения Навье-Стокса (2D, несжимаемая жидкость)

## Уравнение неразрывности

```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
```

## X-импульс

```math
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
```

## Y-импульс

```math
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
```

## Векторная форма

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}
```

```math
\nabla \cdot \mathbf{u} = 0
```

## Стационарная форма (для gPINN)

```math
u u_x + v u_y + \frac{1}{\rho} p_x - \nu (u_{xx} + u_{yy}) = 0
```

```math
u v_x + v v_y + \frac{1}{\rho} p_y - \nu (v_{xx} + v_{yy}) = 0
```

```math
u_x + v_y = 0
```
