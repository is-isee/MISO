# `demo` 共通のルール

- デモアプリのルートディレクトリを `demo/<app_name>` と書く
- 一括実行のためのルール (必須)
  - `cmake -B build && cmake --build build` によりビルドする
  - `./app_run.sh` により、シミュレーションを実行する
  - `./app_clean.sh` により、`demo/<app_name>/build` およびデータを削除する
  - `app_run.sh` および `app_clean.sh` はカレントディレクトリによらず正常動作する
- ファイル整理のための慣習 (推奨)
  - ソースファイルは `demo/<app_name>/src` に置く
  - データディレクトリは `demo/<app_name>` に置く
  - `config.yaml` は `demo/<app_name>` または  `demo/<app_name>/config` に置く
