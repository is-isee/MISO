### フォーマッター

```shell
# CLI
find src include problems tests -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i

# VSCode上の設定 (上書き注意)
mkdir -p .vscode && cp -i vscode/setting.json .vscode/setting.json
```

自動フォーマットしたくない範囲は `// clang-format off` と `// clang-format on` で囲む(`off`や`on`の後にスペースが挿入されていると動かないので注意)。

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

C++のAPIドキュメントをdoxygenで、pythonのAPIドキュメントをsphinxで生成する。全体のフォーマットを揃えるためにdoxygendenではxml形式で出力し、sphinxで取り込む。

sphinx + breathe + exhaleを利用

以下のパッケージをインストール
```shell
pip install sphinx sphinx-rtd-theme sphinx-automodapi sphinx-multiversion breathe exhale sphinx-copybutton
```

```shell
cd docs
make
```

これで`docs/build/index.html`にHTMLが生成される。

## Dev Container

CUDA対応のコードを開発する場合は、Dev Containerを利用することを推奨。
コマンドパレットから `Dev Containers: Reopen in Container` を選択。

`devcontainer`ディレクトリの`devcontainer.json`、`Dockerfile`に設定例あり。
コンテナ実行環境の違いにより微妙に設定が変わるので注意。

ホスト側の要件 (Docker on Linuxの場合):

- Docker
- NVIDIA Driver
- NVIDIA Container Toolkit
