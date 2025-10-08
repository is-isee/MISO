# Quick Start

本セクションでは、MISOのインストール方法と使用方法の概要を説明します。

## Installation


MISOはGitHub (https://github.com/is-isee/MISO) で入手できます。リポジトリをクローンするには、次のコマンドを使用します。

```shell
git clone https://github.com/is-isee/MISO.git
```

MISOのC++プログラムのコンパイルはCMakeを用います。


```shell
cd MISO
cmake -B build -S . -DUSE_CUDA=OFF
cd build
make -j 4
```

これで、CPU版のすべてのプログラムがコンパイルされる。GPU版をコンパイルする場合は、`-DUSE_CUDA=ON`を指定してください。

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

インストールの後、 `build` ディレクトリに `problem` ディレクトリで定義された複数のターゲットが利用可能になります。例えば、2D Orszag-Tang渦問題を実行するには、 `build` ディレクトリで次のコマンドを使用します。

```shell
cd build
./mhd_vortex_2d
```

なお、 `build` ディレクトリで `make` を実行すると、すべてのターゲットがコンパイルされますが、特定のターゲットのみをコンパイルすることも可能です。

```shell
cd build
make -j 4 mhd_vortex_2d
```

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

シミュレーションパラメータは、 `problems/mhd_vortex_2d/config.yaml` ファイルを編集することで制御できます。例えば、格子点数は次のように変更できます。

```yaml
grid:
    i_size: 256
    j_size: 256
```      

GPU版の実行には、NVIDIA HPC SDKが必要です。CPU版でコンパイル済みの場合は、 `cmake` から再実行する必要があります。

```shell
cd MISO
cmake -B build -S . -DUSE_CUDA=ON
cd build
make -j 4 mhd_vortex_2d
mpirun -n 1 --bind-to none   --mca pml ob1   --mca btl tcp,self,vader   --mca coll ^hcoll   --mca osc ^ucx   ./mhd_vortex_2d
```

GPU版では、たとえ1つのプロセスを使用する場合でも、プログラムを実行するには `mpirun` コマンドを使用する必要があります。

一方、GPU版で複数のプロセスを使用する場合は、さらに `--map-by ppr:N:node` オプションを指定する必要があります (Nは1ノードあたりのプロセス数)。例えば、4プロセスで実行する場合は、次のようになります。

```shell
mpirun -n 4   --map-by ppr:3:node --bind-to none   -x CUDA_DEVICE_ORDER=PCI_BUS_ID   -x CUDA_VISIBLE_DEVICES=0,1   --mca pml ob1 --mca btl tcp,self,vader --mca coll ^hcoll --mca osc ^ucx   ./mhd_vortex_2d
```

シミュレーションの出力は `problems/mhd_vortex_2d/data` ディレクトリに保存されます。途中で計算を停止した場合でも、保存されたデータから再開されます。計算を最初からやり直すには、 `problems/mhd_vortex_2d/data` ディレクトリを削除してください。以下のようにしてもデータを削除することができます。

```shell
make clean_data_mhd_vortex_2d
```