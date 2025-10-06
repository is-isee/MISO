Test Problems
=============

This section describes the test problems available in MISO. Each problem has its own directory under the ``problems/`` directory, which contains the main code file ( ``main.cpp`` for CPU version and ``main.cu`` for GPU version) and configuration files such as ``config.yaml``.

In some cases, additional files like ``custom_boundary_condition.cpp`` and ``force.cpp`` are included to set more customized boundary conditions and external forces, as explained in the :doc:`../customization/index` section.

本セクションでは、MISOで利用可能なテスト問題について説明します。 ``problems/`` ディレクトリ以下に、各問題のディレクトリが設置してあり、 ``main.cpp`` (GPU版では ``main.cu`` )、および ``config.yaml`` などの設定ファイルが含まれています。

:doc:`../customization/index` で説明するように追加で ``custom_boundary_condition.cpp`` や ``force.cpp``  を設定することでよりカスタマイズした境界条件や外力を設定している場合もあります。

.. toctree::
   :maxdepth: 1
   :caption: Problems:
   
   hd_shock_tube_1d.md
   mhd_shock_tube_1d.md
   mhd_vortex.md
   kelvin_helmholtz.md
   rayleigh_taylor.md
   geomagnetosphere.md
   