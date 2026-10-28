# input_data

MISO のデモやテストが読む入力データを置く (計算の出力先 `data/` とは別。`data*/` は .gitignore で無視される)。

一度置いたファイルは書き換えない。作り直したときは別の名前で置く (git の履歴が大きくならないように、また UUID とファイルの対応を変えないため)。

| ファイル | 中身 | 読む C++ |
|---|---|---|
| `eos_table_roen_blend.tbl` | 状態方程式と不透明度の表 (ln ρ, ln e の 2 次元) | `miso::eos::TableEOS` |
| `stratification_miso.tbl` | 太陽の 1 次元成層 (高さ z の関数) | `miso::Stratification` |

どちらも堀田の非公開リポジトリ R2D2plus-input で作り、`contracts/export_public.py` でヘッダから来歴 (作成した機械のパスなど) を除いた。MISO には出来上がったファイルだけを置く。作り方や中身について質問があれば堀田に聞く。

## r2d2plus-table 形式

R2D2plus と共通の、1 ファイルで自己記述する表の形式。テキストのヘッダの後に配列が続く。

```text
r2d2plus-table 1                          1 行目は必ずこれ
kind stratification_z                     表の種類
uuid a67daa42-057d-4376-b602-d267b02395fe 表の名札 (配列のビットが変わったら新しくする)
layout row_major little_endian float64    配列の並び (これ以外はない)
binary_offset 4096                        配列の始まる位置 [バイト]
array z 1782                              配列の名前と大きさ (1 次元)
array ro 1782                             2 次元なら "array pr 256 256"
...
meta {"target":"MISO", ...}               作り方の記録 (1 行の JSON)
end
```

- `meta` は `end` の直前の行。ヘッダの後は `binary_offset` バイト目まで改行で埋めてある。
- ファイルの先頭から数えて `binary_offset` バイト目から、`array` の行の順に配列が隙間なく並ぶ。値は little-endian の float64。
- 2 次元の配列は行優先 (最後の添字が速い)。
- `meta` の `parents` に元にした表の UUID、`sources` に出典、`source_sha256` に来歴を除く前のファイルの sha256 がある。

Python (numpy) で読む例:

```python
import json
import numpy as np


def read_table(path):
    with open(path, "rb") as f:
        head = f.read(65536).decode("utf-8", errors="replace")
    header, shapes = {}, {}
    for line in head.splitlines()[1:]:
        if line == "end":
            break
        key, _, rest = line.partition(" ")
        if key == "array":
            name, *shape = rest.split()
            shapes[name] = tuple(int(n) for n in shape)
        elif key == "meta":
            header["meta"] = json.loads(rest)
        elif key:
            header[key] = rest
    arrays, offset = {}, int(header["binary_offset"])
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        arrays[name] = np.fromfile(path, dtype="<f8", count=count, offset=offset)
        arrays[name] = arrays[name].reshape(shape)
        offset += 8 * count
    return arrays, header


arrays, header = read_table("input_data/stratification_miso.tbl")
print(header["kind"], arrays["z"][:3])
```

## eos_table_roen_blend.tbl

太陽光球の計算 (#201) に使う、状態方程式と不透明度の表。

- 作成: R2D2plus-input の `stratification_eos/eos_miso.py` と `blend_eos.py --kind roen`
- UUID: `7010569f-038f-46f2-8f58-3c351dba80f6` (ファイルの先頭のヘッダにも書いてある)
- sha256: `84714d482cfd924dd9acb7ce0e1741a638aee0041561568d520f3333bd75c612`
- 来歴を除く前の表とは、配列がビット同一で UUID も同じ。その表の sha256 をヘッダの `source_sha256` に書いてある。
- 組成: 標準太陽モデル Model S (Christensen-Dalsgaard et al. 1996, Science 272, 1286) の表面の値 X = 0.73725, Z = 0.019628
- 出典:
  - 状態方程式: FreeEOS (Irwin 2012, <http://freeeos.sourceforge.net/>)
  - 不透明度 (Rosseland 平均): OPAL (Iglesias & Rogers 1996, ApJ 464, 943)。
    OPAL の範囲外 (主に低温側) は Ferguson et al. (2005, ApJ 623, 585) で埋めた。
    ヘッダの `recipe` と `parents` は、この継ぎ方 (OPAL だけの表と Ferguson を含む表の UUID) を記録したもの。

### 中身

形式は下の「r2d2plus-table 形式」。2 次元の量は `[ro, en]` の順の行優先 (en が連続)。C++ では `miso::eos::TableEOS` (`miso/include/miso/eos_table.hpp`) で読む。

| 名前 | 大きさ | 中身 |
|---|---|---|
| `ro` | 256 | ln(ρ [g/cm³])、一様。ρ = 1e-13〜1 |
| `en` | 256 | ln(e [erg/g])、一様。e は単位質量あたりの内部エネルギー。e = 5e11〜1e15 |
| `pr` | 256×256 | 圧力 [dyn/cm²] |
| `te` | 256×256 | 温度 [K] |
| `se` | 256×256 | 比エントロピー [erg/g/K] |
| `cs` | 256×256 | 断熱音速 [cm/s] |
| `op` | 256×256 | Rosseland 平均不透明度 [cm²/g] |
| `dprdro` | 256×256 | (∂p/∂ρ)_e |
| `dprden` | 256×256 | (∂p/∂e)_ρ |
| `valid` | 256×256 | 1: 温度 1e3〜1e8 K の範囲で解けた点、0: 範囲外 (他の量は 0) |

注意:

- `op` は、低密度・高温の一部 (ρ ≲ 3.4e-7 g/cm³ かつ T ≈ 3e4〜5e6 K、OPAL の定義域 log R ≥ -8 の外) で `inf` になっている。
  光球 (T ≲ 2e4 K) の計算では届かないが、読み込む側は `inf` を検出できるようにしておく。
  上端を彩層・コロナへ伸ばすときは、輻射輸送を切るなどの扱いが要る。
- 作成時に Model S の成層 (r > 0.9 R☉) をこの表から双一次内挿して比べたところ、圧力と温度の差は最大 1.4% (水素の電離が急な光球付近) だった。

## stratification_miso.tbl

太陽光球の計算 (#201) の初期条件に使う、1 次元の成層。R2D2plus の光球の計算と同じ成層である。

- 作成: R2D2plus-input の `stratification_eos/stratification_miso.py`
- UUID: `a67daa42-057d-4376-b602-d267b02395fe`
- sha256: `cb337fbbc6e8a41fc20060380c97c9973d1707594811340b3afb0385503ef0ef`
- 元: 標準太陽モデル Model S を FreeEOS で静水圧平衡に積み直し、表面より上へ延長したもの。使うときは Model S を引用する:
  Christensen-Dalsgaard, J., et al. 1996, "The Current State of Solar Modeling", Science, 272, 1286
  (<https://ui.adsabs.harvard.edu/abs/1996Sci...272.1286C/abstract>)
  点は元の格子のまま (z は一様でない。表面付近の間隔は 1〜7 km)。
  `eos_table_roen_blend.tbl` の (ρ, e) の範囲に入る点だけを残した (z = -322 Mm 〜 +2.24 Mm、1782 点)。
- 組成は `eos_table_roen_blend.tbl` と同じ。

### 中身

すべて 1 次元で、z の小さい順。単位は CGS。EoS の表と違い、`ro` と `en` は対数でない。C++ では `miso::Stratification` (`miso/include/miso/stratification.hpp`) で読む。

| 名前 | 中身 |
|---|---|
| `z` | 高さ r − R☉ [cm]。上向きが正、0 が太陽半径 R☉ = 6.9599e10 cm (光球の底の目安、T = 5850 K) |
| `ro` | 密度 [g/cm³] |
| `en` | 単位質量あたりの内部エネルギー [erg/g] |
| `pr` | 圧力 [dyn/cm²] |
| `te` | 温度 [K] |
| `se` | 比エントロピー [erg/g/K] |
| `gx` | 重力加速度の大きさ [cm/s²]。向きは −z。球の重力なので z で変わる (表の下端では表面の約 3 倍) |

注意:

- z > 0 (Model S の外側) は、R2D2plus-input が延長した部分で、観測される彩層ではない。
- この (ρ, e) から `eos_table_roen_blend.tbl` で引いた圧力は、`pr` と z ≤ 2 Mm で 0.4% 以内、上端 (2.24 Mm) まで含めると 1.4% 以内で合う
  (表の内挿の誤差)。だから MISO で `ro` と `en` から始めると、静水圧平衡から少しずれる (#201 で、熱対流が起これば変わるので気にしないと決めた)。

