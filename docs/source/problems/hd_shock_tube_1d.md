# 1D Hydrodynamic Shock Tube Problem

This is one-dimensional hydrodynamic shock tube problem {cite}`sod_1978JCoPh..27....1S` , in which a discontinuity in the initial condition generates a shock wave, a contact discontinuity, and a rarefaction wave. This problem is often used to validate numerical schemes for solving hydrodynamic equations.

In the test problems provided by MISO, we provide three one-dimesnaional problems with $x$, $y$, and $z$ axes. In this page, we explain the problem along the $x$ axis.

1次元の衝撃波管問題 {cite}`sod_1978JCoPh..27....1S` です。初期条件に不連続があるため、衝撃波、接触不連続、希薄波が発生します。この問題は、流体の方程式を解く数値スキームの検証によく用いられます。

MISOで提供しているテスト問題では、$x$, $y$, $z$の各方向に1次元問題を実施しているが、本ページでは、$x$方向の問題を説明する。


## Location

The problem is available at `problems/hd_shock_tube_1d/`

## Geometry

The geometry extends $0 \leq x \leq 1$.

## Initial Conditions

The initial condition is described as the left and right states separated at $x=0.5$ as follows. The ratio of specific heats is set to $\gamma = 1.4$.

初期条件は、$x=0.5$で分離された左側と右側の状態で記述されます。比熱比は$\gamma = 1.4$とします。

$$
\begin{align*}
\begin{pmatrix}
\rho_\mathrm{L} \\
p_\mathrm{L} \\
v_\mathrm{L}
\end{pmatrix}
&=
\begin{pmatrix}
1.0 \\
1.0 \\
0.0 
\end{pmatrix} \\
\begin{pmatrix}
\rho_\mathrm{R} \\
p_\mathrm{R} \\
v_\mathrm{R}
\end{pmatrix}
&=
\begin{pmatrix}
0.125 \\
0.1 \\
0.0
\end{pmatrix}
\end{align*}
$$



## Boundary Conditions

We set symmetric boundary conditions for all physical quantities. In `config_x.yaml`, it is set as follows. The configure files for the $y$ and $z$ directions are available at `config_y.yaml` and `config_z.yaml`, respectively.

すべての物理量について、対称境界条件を設定します。`config_x.yaml`で以下のように設定してあります。$y$方向と$z$方向の設定ファイルは、それぞれ`config_y.yaml`と`config_z.yaml`にあります。

```yaml
# config_x.yaml
boundary_condition:
  # please use "standard" or "custom" for boundary_type
  boundary_type: standard

  periodic:
    x: false
    y: false
    z: false

  ro:
    x: [symmetric, symmetric]
    y: [symmetric, symmetric]
    z: [symmetric, symmetric]

    ...
```

## Results

You can run a python program to generate a result plot and compare the results with different directions, i.e., $x$, $y$, and $z$ directions. A result plot is available at `py/problems/figs/hd_shock_tube_1d.png`.

pythonプログラムを実行して、結果のプロットを生成し、$x$, $y$, $z$方向の結果を比較できます。結果のプロットは `py/problems/figs/hd_shock_tube_1d.png` にあります。

```shell
cd py/problems/
python hd_shock_tube_1d.py
```

The python program calculates the analytical solution and you can compare the numerical solution with the analytical solution.

本pythonプログラムでは、解析解も計算しており、数値解と解析解の比較も行っています。

![hd_shock_tube_result](../_static/images/hd_shock_tube_1d.png)