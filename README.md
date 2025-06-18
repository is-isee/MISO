# MISO　(Mhd ISee Open source code)
## 実行方法

CMakeを用いたコンパイル。`CMakeLists.txt`に設定あり。

### problems/XXX/
#### CPU版
```shell
cmake -B build -S . -DUSE_CUDA=OFF # GPU版の時は-DUSE_CUDA=ON
cd build
make mhd_shock_tube_1d # problems/以下にある課題名を指定
./mhd_shock_tube_1d # makeのtargetと同じ名前の実行ファイルが生成される
```

#### GPU版
NVIDIA HPC SDKが必要
```shell
cmake -B build -S . -DUSE_CUDA=OFF # GPU版の時は-DUSE_CUDA=ON
cd build
make mhd_shock_tube_1d # problems/以下にある課題名を指定
mpirun -n 1   --bind-to none   --mca pml ob1   --mca btl tcp,self,vader   --mca coll ^hcoll   --mca osc ^ucx   ./mhd_shock_tube_1d  # makeのtargetと同じ名前の実行ファイルが生成される
```

### unit_test
```shell
cmake -B build -S . -DUSE_CUDA=OFF # unit_testはCPU版のみ
cd build
make unit_test
./unit_test
```

## python

シミュレーションデータ読み込みのためのpythonライブラリ`pyMISO`。以下でインストール
```shell
pip install .
```

使用例
```python
import pyMISO
data_dir = './data' # data directoryを指定
d = pyMISO.Data(data_dir) # pyMISO.Dataオブジェクト生成
```
## ドキュメント生成

Doxygenを利用

```shell
git submodule update --init --recursive
mkdir -p docs/doxygen
doxygen Doxyfile
```

`docs/doxygen/html/index.html`にHTMLが生成される。