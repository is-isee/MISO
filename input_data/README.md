# input_data

MISO のデモやテストが読む入力データを置く (計算の出力先 `data/` とは別。`data*/` は .gitignore で無視される)。

一度置いたファイルは書き換えない。作り直したときは別の名前で置く (git の履歴が大きくならないように、また UUID とファイルの対応を変えないため)。

## eos_table_roen_blend.tbl

太陽光球の計算 (#201) に使う、状態方程式と不透明度の表。

- 作成: 堀田の非公開リポジトリ R2D2plus-input の `stratification_eos/eos_miso.py` と `blend_eos.py --kind roen` で作り、
  `contracts/export_public.py` でヘッダから来歴 (作成した機械のパスなど) を除いた。MISO には出来上がった表だけを置く。
  作り方や中身について質問があれば堀田に聞く。
- UUID: `7010569f-038f-46f2-8f58-3c351dba80f6` (ファイルの先頭のヘッダにも書いてある)
- sha256: `84714d482cfd924dd9acb7ce0e1741a638aee0041561568d520f3333bd75c612`
- 来歴を除く前の表とは、配列がビット同一で UUID も同じ。その表の sha256 をヘッダの `source_sha256` に書いてある。
- 組成: 標準太陽モデル Model S (Christensen-Dalsgaard et al. 1996, Science 272, 1286) の表面の値 X = 0.73725, Z = 0.019628
- 出典:
  - 状態方程式: FreeEOS (Irwin 2012, <http://freeeos.sourceforge.net/>)
  - 不透明度 (Rosseland 平均): OPAL (Iglesias & Rogers 1996, ApJ 464, 943)。
    OPAL の範囲外 (主に低温側) は Ferguson et al. (2005, ApJ 623, 585) で埋めた。
    ヘッダの `recipe` と `parents` は、この継ぎ方 (OPAL だけの表と Ferguson を含む表の UUID) を記録したもの。

### 形式

r2d2plus-table 形式。テキストのヘッダ (`binary_offset` バイト) の後に、little-endian の float64 が `array` 宣言の順に並ぶ。2 次元の量は `[ro, en]` の順の行優先 (en が連続)。

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
