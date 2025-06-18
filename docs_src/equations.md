\page equations Equations
In vector form:

\f[
\begin{align}
    \frac{\partial \rho}{\partial t} =& - \nabla\cdot \left(\rho \boldsymbol{v}\right) \\
    \frac{\partial }{\partial t}\left(\rho \boldsymbol{v}\right) =& 
    -\nabla\cdot\left(\rho\boldsymbol{vv}\right) - \nabla p -\nabla\left(\frac{B^2}{8\pi}\right)
    +\nabla\cdot\left(\frac{\boldsymbol{BB}}{4\pi}\right) \\
    \frac{\partial \boldsymbol{B}}{\partial t} =&
    \nabla\times\left(\boldsymbol{v}\times\boldsymbol{B}\right) - \nabla \psi \\
    \frac{\partial \psi}{\partial t} &= - \nabla\cdot\left(c^2_h \boldsymbol{B}\right)
    - \frac{\psi}{\tau}
    \\
    \frac{\partial }{\partial t}\left(\rho e_\mathrm{int} + \frac{1}{2}\rho v^2 \frac{B^2}{8\pi}\right)=&
    - \nabla\cdot\left[\left(\rho e_\mathrm{int} + p + \frac{1}{2}\rho v^2 + \frac{B^2}{4\pi}\right)\boldsymbol{v} - \frac{(\boldsymbol{v}\cdot \boldsymbol{B})\boldsymbol{B}}{4\pi}\right] \\
    &-\frac{1}{4\pi}\boldsymbol{B}\cdot \left(\nabla \psi\right)
\end{align}
\f]

We define the total energy per unit volume \f$E_\mathrm{total}\f$ and \f$H\f$ (enthalpy + kinetic energy + magnetic energy + magnetic pressure) as:

\f[
    \begin{align}
    H =& \rho e_\mathrm{int} + p + \frac{1}{2}\rho v^2 + \frac{B^2}{4\pi}
    \end{align}
\f]

In fully explicit form:

\f[
    \begin{align}
    \frac{\partial \rho}{\partial t} =&
        - \frac{\partial \rho v_x}{\partial x}
        - \frac{\partial \rho v_y}{\partial y}
        - \frac{\partial \rho v_z}{\partial z} \\
    \frac{\partial \rho v_x}{\partial t} =&
        - \frac{\partial}{\partial x}(\rho v_x v_x)
        - \frac{\partial}{\partial y}(\rho v_x v_y)
        - \frac{\partial}{\partial z}(\rho v_x v_z) \\
       &- \frac{\partial p}{\partial x}
        - \frac{\partial}{\partial x}\left(\frac{B^2}{8\pi}\right) \\
       &+ \frac{1}{4\pi} \left[
          \frac{\partial}{\partial x}(B_x B_x)
        + \frac{\partial}{\partial y}(B_x B_y)
        + \frac{\partial}{\partial z}(B_x B_z)
        \right]
         \\
    \frac{\partial \rho v_y}{\partial t} =&
        - \frac{\partial}{\partial x}(\rho v_y v_x)
        - \frac{\partial}{\partial y}(\rho v_y v_y)
        - \frac{\partial}{\partial z}(\rho v_y v_z) \\
       &- \frac{\partial p}{\partial y}
        - \frac{\partial}{\partial y}\left(\frac{B^2}{8\pi}\right) \\
       &+ \frac{1}{4\pi} \left[
          \frac{\partial}{\partial x}(B_y B_x)
        + \frac{\partial}{\partial y}(B_y B_y)
        + \frac{\partial}{\partial z}(B_y B_z)
        \right]
         \\
    \frac{\partial \rho v_z}{\partial t} =&
        - \frac{\partial}{\partial x}(\rho v_z v_x)
        - \frac{\partial}{\partial y}(\rho v_z v_y)
        - \frac{\partial}{\partial z}(\rho v_z v_z) \\
       &- \frac{\partial p}{\partial z}
        - \frac{\partial}{\partial z}\left(\frac{B^2}{8\pi}\right) \\
       &+ \frac{1}{4\pi} \left[
          \frac{\partial}{\partial x}(B_z B_x)
        + \frac{\partial}{\partial y}(B_z B_y)
        + \frac{\partial}{\partial z}(B_z B_z)
        \right]
         \\
    \frac{\partial B_x}{\partial t} = &
    - \frac{\partial}{\partial y}\left(v_y B_x\right)
    + \frac{\partial}{\partial y}\left(v_x B_y \right)
    - \frac{\partial}{\partial z}\left(v_z B_x\right)
    + \frac{\partial}{\partial z}\left(v_x B_z \right)
    - \frac{\partial \psi}{\partial x}
         \\
    \frac{\partial B_y}{\partial t} = &
    - \frac{\partial}{\partial x}\left(v_x B_y\right)
    + \frac{\partial}{\partial x}\left(v_y B_x \right)
    - \frac{\partial}{\partial z}\left(v_z B_y\right)
    + \frac{\partial}{\partial z}\left(v_y B_z \right)
    - \frac{\partial \psi}{\partial y}
         \\
    \frac{\partial B_z}{\partial t} = &
    - \frac{\partial}{\partial x}\left(v_x B_z\right)
    + \frac{\partial}{\partial x}\left(v_z B_x \right)
    - \frac{\partial}{\partial y}\left(v_y B_z\right)
    + \frac{\partial}{\partial y}\left(v_z B_y \right)
    - \frac{\partial \psi}{\partial z}
         \\
    \frac{\partial \psi}{\partial t} =&
        - c_h^2 \left(
        \frac{\partial B_x}{\partial x} + \frac{\partial B_y}{\partial y} + \frac{\partial B_z}{\partial z}
        \right)
        - \frac{\psi}{\tau}
         \\
    \frac{\partial}{\partial t}
    \left(
        \rho e_\mathrm{int} + \frac{1}{2}\rho v^2 + \frac{B^2}{8\pi}
    \right) =&
    - 
    -\frac{\partial }{\partial x}\left(H v_x\right) 
    + \frac{1}{4\pi}\frac{\partial}{\partial x}
    \left[\left(\boldsymbol{v} \cdot\boldsymbol{B}\right)B_x\right]
    - \frac{1}{4\pi}B_x\frac{\partial \psi}{\partial x} \\
   &-\frac{\partial }{\partial y}\left(H v_y\right) 
    + \frac{1}{4\pi}\frac{\partial}{\partial y}
    \left[\left(\boldsymbol{v} \cdot\boldsymbol{B}\right)B_y\right]
    - \frac{1}{4\pi}B_y\frac{\partial \psi}{\partial y} \\
   &-\frac{\partial }{\partial z}\left(H v_z\right) 
    + \frac{1}{4\pi}\frac{\partial}{\partial z}
    \left[\left(\boldsymbol{v} \cdot\boldsymbol{B}\right)B_z\right]
    - \frac{1}{4\pi}B_z\frac{\partial \psi}{\partial z}
    \end{align}
\f]