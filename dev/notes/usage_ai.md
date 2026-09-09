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
find miso demo -type d -name build -prune -o -type f \
  \( -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) \
  -print0 | xargs -0 clang-format --dry-run --Werror -fallback-style=none
ruff check pymiso && ruff format pymiso --check
pytest pymiso/tests   # python -m pytest はリポジトリ直下の古い pymiso/__init__.py を拾うので使わない
```

## コードの約束

- C++: `.clang-format` (LLVM ベース, 2 スペース, 82 桁)。CUDA と共用するコードは `__host__ __device__` を付け、`MISO_LAMBDA` を使う。`Real` 型の演算に double リテラルを混ぜない (`Real(0.5)` と書く)。
- `MISO_LAMBDA` の中でクラスのメンバ変数を直接使わない。メンバ変数は `this` (ホスト側のポインタ) 経由でキャプチャされ、GPU で不正アクセスになる (#170)。ラムダの直前でローカル変数にコピーしてから使う。

  ```cpp
  const Real gm_ = gm;  // メンバ変数をローカルにコピー
  for_each(btag, range, MISO_LAMBDA(int i) { pr[i] = (gm_ - Real(1)) * qq.ro[i] * qq.ei[i]; });
  ```

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

すべて `io.save_dir` (既定 `data/`) の下に書かれる。読み込みは `pymiso.Data`。

- `save_dir/config.yaml`: デフォルト値を補完した設定
- `save_dir/grid.bin`: 全体格子の座標 (先頭 uint32 の要素サイズ + x, y, z)
- `save_dir/mpi/coords.csv`: ランクごとの MPI 座標
- `save_dir/time/time.NNNNNNNN.txt`: 各出力の時刻・出力番号・ステップ数
- `save_dir/time/n_output.txt`: 最新の出力番号 (`pymiso.Time` が最初に読む)
- `save_dir/mhd/mhd.<n_output>.<rank>.bin`: ランクごとの MHD 変数 (先頭 uint32 の要素サイズ + 9 変数、ゴーストセル込み)
- `save_dir/rt/rank_<rank>.bin`: 輻射輸送の出力
