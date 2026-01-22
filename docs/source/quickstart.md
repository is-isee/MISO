# Quick Start

本セクションでは、MISOのインストール方法と使用方法の概要を説明します。

## Installation

MISOはGitHub (<https://github.com/is-isee/MISO>) で入手できます。リポジトリをクローンするには、次のコマンドを使用します。

```shell
git clone https://github.com/is-isee/MISO.git
```

## Usage (CPU version)

MISOのコンパイルにはCMakeを用います。
例えば、2D Orszag-Tang渦問題を実行してみましょう。

```shell
cd demo/mhd2d_vortex
cmake -B build -DUSE_CUDA=OFF
make -C build
```

これで、実行ファイル build/mhd2d_vortex が生成されます。

2D Orszag-Tang渦問題は以下のように実行することができます。

```shell
mpiexec -n 2 ./build/mhd2d_vortex
```

MPIを使用している場合、適切な環境変数・オプションを要求されることがあります。OpenMPIを使用している場合には、自動で環境変数を設定したうえで上記のコマンドを実行するシェルスクリプトが demo/mhd2d_vortex/app_run.sh に用意されています。

```shell
./app_run.sh
```

demo/mhd2d_vortex/data にシミュレーション結果が保存されます。
途中で計算を停止した場合でも、保存されたデータから再開されます。計算を最初からやり直すには、 problems/mhd_vortex_2d/data/ ディレクトリを削除してください。以下のようにしてもデータを削除することができます。

```shell
make clean_data_mhd_vortex_2d
```

シミュレーションパラメータは、 demo/mhd2d_vortex/config.yaml ファイルを編集することで制御できます。例えば、格子点数は次のように変更できます。

```yaml
grid:
    i_size: 256
    j_size: 256
```

## Usage (CUDA version)

CUDA版の実行には、NVIDIA HPC SDKが必要です。
CUDA版をコンパイルする場合は、CMakeのオプションで `-DUSE_CUDA=ON` を指定します。

```shell
cd demo/mhd2d_vortex
cmake -B build -DUSE_CUDA=ON
make -C build
```

実行ファイル build/mhd2d_vortex やその実行方法はCPU版と同様です。
実行時に必要なオプションや環境変数の設定方法が環境により変わる場合があるため、以下は例として参考にしてください。

```shell
mpiexec -n 2 --bind-to none --mca pml ob1 --mca btl tcp,self,vader --mca coll ^hcoll --mca osc ^ucx ./build/mhd2d_vortex
```

GPU版で複数のプロセスを使用する場合は、オプションとして "--map-by ppr:N:node" (N は 1 ノードあたりのプロセス数) や "-x CUDA_VISIBLE_DEVICES=M,L,..." (M, L, ... は各ノードで利用可能とする CUDA デバイス番号) などを指定することを推奨します。例えば、 2 ノード、ノード当たり 2 プロセス、各プロセスが 1 CUDA デバイスを使用し、合計 4 プロセスで実行する場合は、次のようになります。

```shell
mpiexec -n 4 --map-by ppr:2:node --bind-to none -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x CUDA_VISIBLE_DEVICES=0,1 --mca pml ob1 --mca btl tcp,self,vader --mca coll ^hcoll --mca osc ^ucx ./build/mhd2d_vortex
```

## Python module pymiso

MISOでは、解析のためのPythonモジュールを提供しています。リポジトリのルートディレクトリで、以下のようにしてインストールしてください。

```shell
# for normal users
pip install -e ".[vis]"

# minimum install
pip install -e .

# for development
pip install -e ".[vis,dev,docs]"
```

シミュレーション結果は以下のように読み込むことができます。

```python
import pymiso
import matplotlib.pyplot as plt

# relative path to the data directory
d = pymiso.Data("./demo/mhd2d_vortex/data")

n_step = 80
d.load(n_step)
plt.pcolormesh(d.x, d.y, d.ro)
plt.show()
```

各デモ問題には可視化用のスクリプトが含まれています。例えば、2D Orszag-Tang渦問題の結果を可視化するには、 demo/mhd2d_vortex/plot_data.py を実行します。

```shell
cd demo/mhd2d_vortex
python plot_data.py
```

実行後、シミュレーション結果の画像ファイルが demo/mhd2d_vortex/figs に保存されます。
