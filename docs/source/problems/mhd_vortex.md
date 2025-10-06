# Orszag-Tang Vortex Problem

This is the Orszag-Tang Vortex Problem {cite}`orszag_1979JFM....90..129O` which is a standard test problem for magnetohydrodynamics (MHD) codes. The problem involves the interaction of two vortex flows in a magnetized medium, leading to the formation of complex structures and turbulence.

Orszag-Tang渦問題 {cite}`orszag_1979JFM....90..129O` は、磁気流体力学（MHD）コードの標準的なテスト問題です。この問題は、磁化された媒体における2つの渦流の相互作用を含み、複雑な構造と乱流の形成をもたらします。

## Location

The problem is available at `problems/mhd_vortex_2d/`

## Geometry

The geometry extends $0 \leq x \leq 1$ and $0 \leq y \leq 1$.

## Initial Conditions

The initial condition is described as follow

$$
\begin{align*}
\rho &= 1, \\
p &= \frac{1}{\gamma}, \\
v_x &= -v_0 \sin(2\pi y), \\
v_y & = v_0 \sin(2\pi x), \\
v_z & = 0, \\
B_x & = B_0 \sin(2\pi y),\\
B_y & = B_0 \sin(4\pi x),\\
B_z & = 0,
\end{align*}
$$

where $v_0 = 1$, and $B_0 = \sqrt{4\pi}/\gamma$.

## Boundary Conditions

We set periodic boundary condition on the all boundries for all quantities.

周期境界条件を全ての境界に対して全ての物理量に設定します。

```yaml
# config_x.yaml
boundary_condition:
  # please use "standard" or "custom" for boundary_type
  boundary_type: standard

  periodic:
    x: true
    y: true
    z: true # it does not matter in 2D x-y problems

    ...
```

We note that when the periodic bounday condition flag is set to true, the symmetric boundary condition does not work.

$x$方向の周期境界条件フラグをtrueに設定すると、対称境界条件が機能しなくなることに注意してください。

## Results

You can run a python program `mhd_vortex_2d.py` to generate result plots. The result plots are stored at `py/problems/figs/mhd_vortex_2d/`.

用意されたpythonプログラム `mhd_vortex_2d.py` を実行することにより、結果を結果のプロットは `py/problems/figs/mhd_vortex_2d/` に保存されます。

```shell
cd py/problems/
python mhd_vortex_2d.py
```

![mhd_vortex_2d](../_static/images/mhd_vortex_2d.gif)