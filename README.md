# MISO　(Mhd ISee Open source code)

[![Run tests on CPU](https://github.com/is-isee/MISO/actions/workflows/test_cpu.yml/badge.svg)](https://github.com/is-isee/MISO/actions/workflows/test_cpu.yml)

## 実行方法

CMakeを用いたコンパイル。`CMakeLists.txt`に設定あり。

### 単体テスト

```shell
cmake -B build -S . -DUSE_CUDA=OFF # unit_testはCPU版のみ
cd build
make unit_tests
make test
```

### 典型課題

#### CPU版

```shell
cmake -B build -S . -DUSE_CUDA=OFF # CPU版
cd build
make mhd_shock_tube_1d # problems/以下にある課題名を指定
./mhd_shock_tube_1d # makeのtargetと同じ名前の実行ファイルが生成される
```

#### GPU版

NVIDIA HPC SDKが必要。

```shell
cmake -B build -S . -DUSE_CUDA=ON # GPU版
cd build
make mhd_shock_tube_1d # problems/以下にある課題名を指定
mpirun -n 1   --bind-to none   --mca pml ob1   --mca btl tcp,self,vader   --mca coll ^hcoll   --mca osc ^ucx   ./mhd_shock_tube_1d  # makeのtargetと同じ名前の実行ファイルが生成される
```

### フォーマット

```shell
# CLI
find . -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i

# VSCode上の設定 (上書き注意)
mkdir -p .vscode && cp -i vscode/setting.json .vscode/setting.json
```

## python

シミュレーションデータ読み込みのためのpythonライブラリ`pyMISO`。

### インストール

```shell
# 通常のユーザー
pip install -e ".[vis]"

# 最小インストール (CI等)
pip install -e ".[vis]"

# 開発時
pip install -e ".[all]"
```

### 使用例

```python
import pyMISO
data_dir = './problems/mhd_shock_tube_1d/data_x' # data directoryを指定
d = pyMISO.Data(data_dir) # pyMISO.Dataオブジェクト生成
```

### リンター

```shell
# 検証のみ
ruff check pyMISO

# 自動修正
ruff check pyMISO --fix
```

### フォーマッター

```shell
ruff format pyMISO
```

### 静的型解析

```shell
mypy pyMISO
```

## ドキュメント生成

Doxygenを利用

```shell
git submodule update --init --recursive
mkdir -p docs/doxygen
doxygen Doxyfile_cpu # CPU版 -> docs/doxygen_cpuに生成
doxygen Doxyfile_gpu # GPU版 -> docs/doxygen_gpuに生成
```

`docs/doxygen_[cpu,gpu]/html/index.html`にHTMLが生成される。
