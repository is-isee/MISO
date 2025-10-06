# Quick Start

This section provides a quick overview of how to install and use MISO for your simulations.

本セクションでは、MISOのインストール方法と使用方法の概要を説明します。

## Installation

MISO is available on GitHub (https://github.com/is-isee/MISO). You can clone the repository using the following command:

MISOはGitHub (https://github.com/is-isee/MISO) で入手できます。リポジトリをクローンするには、次のコマンドを使用します。

```shell
git clone https://github.com/is-isee/MISO.git
```

C++ program of MISO is compiled using CMake.

MISOのC++プログラムのコンパイルはCMakeを用います。


```shell
cd MISO
cmake -B build -S . -DUSE_CUDA=OFF
cd build
make -j 4
```

All CPU version programs will be compiled. If you want to compile the GPU version, specify `-DUSE_CUDA=ON`.

これで、CPU版のすべてのプログラムがコンパイルされる。GPU版をコンパイルする場合は、`-DUSE_CUDA=ON`を指定してください。

MISO also provides Python modules for analysis. You can install them as follows.

MISOでは、解析のためのPythonモジュールも提供しています。以下のようにしてインストールしてください。

```shell
# for normal users
pip install -e ".[vis]"

# minimum install (e.g., CI)
pip install -e .

# for development
pip install -e ".[all]"
```

## Usage

After installation, several targets are available in the `build` directory, defined in `problem` directory. For example, to run the 2D Orszag-Tang vortex problem, use the following commands at `build` directory.

インストールの後、 `build` ディレクトリに `problem` ディレクトリで定義された複数のターゲットが利用可能になります。例えば、2D Orszag-Tang渦問題を実行するには、 `build` ディレクトリで次のコマンドを使用します。

```shell
cd build
./mhd_vortex_2d
```

We note that by running `make` in the `build` directory, all targets are compiled, but it is also possible to compile only specific targets.

なお、 `build` ディレクトリで `make` を実行すると、すべてのターゲットがコンパイルされますが、特定のターゲットのみをコンパイルすることも可能です。

```shell
cd build
make -j 4 mhd_vortex_2d
```

Then the simulation results is available in `problems/mhd_vortex_2d/data` directory via `pyMISO` module as:

次に、シミュレーション結果は `pyMISO` モジュールを介して `problems/mhd_vortex_2d/data` ディレクトリで利用可能になります。

```python
import pyMISO
import matplotlib.pyplot as plt

d = pyMISO.Data("../problems/mhd_vortex_2d/data")
n_step = 80
d.load(n_step)
plt.pcolormesh(d.x, d.y, d.ro)
plt.show()
```

Simulation parameters can be controlled by editing `problems/mhd_vortex_2d/config.yaml` file. For example, the number of grid points can be changed as follows.

シミュレーションパラメータは、 `problems/mhd_vortex_2d/config.yaml` ファイルを編集することで制御できます。例えば、格子点数は次のように変更できます。


```yaml
grid:
    i_size: 256
    j_size: 256
```      

To run the GPU version, you need to have NVIDIA HPC SDK installed. If you have compiled the CPU version, you need to re-run from `cmake`.

GPU版の実行には、NVIDIA HPC SDKが必要です。CPU版でコンパイル済みの場合は、 `cmake` から再実行する必要があります。

```shell
cd MISO
cmake -B build -S . -DUSE_CUDA=ON
cd build
make -j 4 mhd_vortex_2d
mpirun -n 1 --bind-to none   --mca pml ob1   --mca btl tcp,self,vader   --mca coll ^hcoll   --mca osc ^ucx   ./mhd_vortex_2d
```

In GPU version, you must use `mpirun` command to run the program, even if you use only one process.

GPU版では、たとえ1つのプロセスを使用する場合でも、プログラムを実行するには `mpirun` コマンドを使用する必要があります。

When using multiple processes in the GPU version, you need to specify the `--map-by ppr:N:node` option (N is the number of processes per node). For example, to run with 4 processes, use the following command.

一方、GPU版で複数のプロセスを使用する場合は、さらに `--map-by ppr:N:node` オプションを指定する必要があります (Nは1ノードあたりのプロセス数)。例えば、4プロセスで実行する場合は、次のようになります。

```shell
mpirun -n 4   --map-by ppr:3:node --bind-to none   -x CUDA_DEVICE_ORDER=PCI_BUS_ID   -x CUDA_VISIBLE_DEVICES=0,1   --mca pml ob1 --mca btl tcp,self,vader --mca coll ^hcoll --mca osc ^ucx   ./mhd_vortex_2d
```

Simulation outputs are saved in `problems/mhd_vortex_2d/data` directory. Even if you stop the calculation halfway, it will be resumed from the saved data. To restart the calculation from the beginning, delete the `problems/mhd_vortex_2d/data` directory. You can also delete the data as follows.

シミュレーションの出力は `problems/mhd_vortex_2d/data` ディレクトリに保存されます。途中で計算を停止した場合でも、保存されたデータから再開されます。計算を最初からやり直すには、 `problems/mhd_vortex_2d/data` ディレクトリを削除してください。以下のようにしてもデータを削除することができます。

```shell
make clean_data_mhd_vortex_2d
```