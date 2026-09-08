# MISO — AI エージェント向けプロジェクト情報

MISO (Mhd ISee Open source code) は磁気流体力学 (MHD) シミュレーションの公開コード。
C++17 ヘッダオンリー (`miso/include/miso/`) + CUDA バックエンド + MPI + yaml-cpp。Python 後処理は `pymiso/`。

## 必ず守ること

- 作業前に `dev/notes/rules_ai.md` (AI 利用ルール) を読む。
- `main` には触らない。`git push`、PR 作成、issue コメントは人間の指示があるときだけ。
- 物理・数値スキーム・公開 API・依存ライブラリ・ディレクトリ構成の変更は、issue で合意する前に実装しない。
- `data*/`, `figs/`, `build/` などの生成物ディレクトリを勝手に削除しない。
- commit 前にフォーマッタ (clang-format, ruff) をかける。

## ディレクトリ

- `miso/` コアライブラリ (CMake target `MISO::MISO`)。`miso/tests/serial`, `miso/tests/parallel` は doctest。
- `demo/<app>/` デモ問題。`src/main.cpp`, `config.yaml` (または `config/config_*.yaml`), `app_run.sh`, `app_clean.sh`, `plot_data.py`。共通ルールは `dev/notes/rules_demo.md`。
- `pymiso/src/pymiso/` Python パッケージ。テストは `pymiso/tests/`。
- `dev/notes/` 開発ルール (`rules_all.md`, `rules_demo.md`, `rules_ai.md`)、環境構築 (`setup.md`)。

## ビルドとテスト

```bash
# コアライブラリとユニットテスト (CPU)
cmake -B build -S miso -DMISO_USE_CUDA=OFF && cmake --build build -j
(cd build && OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_rmaps_base_oversubscribe=yes ctest --output-on-failure)

# デモ (例: mhd2d_vortex)
cmake -B demo/mhd2d_vortex/build -S demo/mhd2d_vortex -DUSE_CUDA=OFF && cmake --build demo/mhd2d_vortex/build -j
./demo/mhd2d_vortex/app_run.sh

# CUDA を使う場合は -DMISO_USE_CUDA=ON / -DUSE_CUDA=ON
# フォーマット (build ディレクトリは除外する)
find miso demo -path '*/build' -prune -o \( -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) -print | xargs clang-format --dry-run --Werror -fallback-style=none
ruff check pymiso && ruff format pymiso --check
python -m pytest pymiso/tests
```

## コードの約束

- C++: `.clang-format` (LLVM ベース, 2 スペース, 82 桁)。CUDA と共用するコードは `__host__ __device__` を付け、`MISO_LAMBDA` を使う。`Real` 型の演算に double リテラルを混ぜない (`Real(0.5)` と書く)。
- ヘッダオンリーなので、テンプレートでない自由関数には `inline` を付ける。
- 実行時エラー (ファイルが開けない等) は `assert` ではなく例外で扱う。既定ビルドは Release で `assert` は消える。
- パス結合は `std::filesystem::path` を使う (文字列連結にしない)。
- Python: ruff (line-length 88), `pathlib` を使う。
- シェル: `set -eu`、スクリプト自身の位置からの相対パスで動くようにする (`dev/notes/rules_all.md`)。

## 設定ファイル (`config.yaml`)

- セクション: `io`, `time`, `grid`, `mpi`, `domain`, `mhd`, `eos`, `rt` と問題固有セクション。
- デフォルト値は `miso/include/miso/config_defaults.hpp` にあり、ユーザー設定と `merge_yaml` で統合される。
- `io.save_dir` は config ファイルからの相対パス。

## 出力形式

- `save_dir/config.yaml`, `grid.bin`, `mpi/coords.csv`, `time/time.NNNNNNNN.txt`, `mhd/mhd.<n_output>.<rank>.bin` (先頭 uint32 の要素サイズ + 9 変数、ゴーストセル込み)。読み込みは `pymiso.Data`。
