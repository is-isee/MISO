# 非公開版C++ RMHDコード
## 実行方法

CMakeを用いたコンパイル。`CMakeLists.txt`に設定あり。
```shell
mkdir build
cd build
cmake ..
make
./a.out
```

## Requirements

- `OpenMP`
- `json`

### Mac

```
brew update
brew install libomp
```

# 境界条件の設定方法

## 概要

本コードでは、`config.json` の `boundary_condition` セクションにて、各物理量・各方向（x, y, z）・各端（inner, outer）の境界条件を設定できます。  
また、周期境界を方向単位で一括指定する `"periodic"` フラグもサポートしています。

---

## JSON設定例

```json
"boundary_condition": {
  "periodic": {
    "x": true,
    "y": false,
    "z": false
  },
  "ro": {
    "x": ["symmetric", "symmetric"],
    "y": ["symmetric", "symmetric"],
    "z": ["symmetric", "symmetric"]
  },
  "vx": {
    "y": ["symmetric", "symmetric"],
    "z": ["symmetric", "symmetric"]
  }
}
```


## 設定の意味
	-	"periodic" に true を指定した方向は、すべての物理量で periodic が自動的に適用されます。
	-	"ro" や "vx" のように物理量ごとに "x", "y", "z" の方向指定ができます。
	-	各方向には [inner, outer] の順に境界条件を並べて指定します。

## 境界条件の種類

- `"symmetric"`: 対称境界
- `"antisymmetric"`: 反対称境界
- `"ro_symmetric"`: 密度のファクターをかけた対称境界
- `"ro_antisymmetric"`: 密度のファクターをかけた反対称境界
- `"periodic"`: 周期境界

## 境界条件の選択

`config.json` の `"boundary_type"` を `"standard"` または `"custom"` に設定してください。

- `"standard"`：JSONで指定された方法で自動的に境界条件を適用
- `"custom"`：ユーザーが `custom_boundary_condition.hpp` に実装した関数が呼ばれます