# MISO　(Mhd ISee Open source code)

[![Run tests on CPU](https://github.com/is-isee/MISO/actions/workflows/test_cpu.yml/badge.svg)](https://github.com/is-isee/MISO/actions/workflows/test_cpu.yml)
[![Run tests of Python](https://github.com/is-isee/MISO/actions/workflows/test_python.yml/badge.svg)](https://github.com/is-isee/MISO/actions/workflows/test_python.yml)

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

### フォーマッター

```shell
# CLI
find src include problems -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i

# VSCode上の設定 (上書き注意)
mkdir -p .vscode && cp -i vscode/setting.json .vscode/setting.json
```

自動フォーマットしたくない範囲は `// clang-format off` と `// clang-format on` で囲む。

```c++
template <typename Real>
inline Real space_centered_4th(const Array3D<Real> &qq, Real dxyzi, int i, int j,
                               int k, int is, int js, int ks) {
  // clang-format off
  return (
        - qq(i + 2 * is, j + 2 * js, k + 2 * ks)
        + 8.0 * qq(i + is, j + js, k + ks)
        - 8.0 * qq(i - is, j - js, k - ks)
        + qq(i - 2 * is, j - 2 * js, k - 2 * ks)
      ) * inv12<Real> * dxyzi;
  // clang-format on
};
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
# 自動修正
ruff check pyMISO --fix

# 検証のみ
ruff check pyMISO
```

### フォーマッター

```shell
# 自動修正
ruff format pyMISO

# 検証のみ
ruff format pyMISO --check
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
