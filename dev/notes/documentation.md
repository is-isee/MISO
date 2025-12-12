# ドキュメント生成

C++のAPIドキュメントをdoxygenで、pythonのAPIドキュメントをsphinxで生成する。全体のフォーマットを揃えるためにdoxygendenではxml形式で出力し、sphinxで取り込む。

sphinx + breathe + exhaleを利用

以下のパッケージをインストール

```shell
sudo apt update && sudo apt install -y doxygen
pip install sphinx sphinx-rtd-theme sphinx-automodapi sphinx-multiversion breathe exhale sphinx-copybutton sphinxcontrib-bibtex myst-parser 
```

```shell
cd docs
make html
```

これで`docs/build/index.html`にHTMLが生成される。

sphinxcontrib-bibtexを利用して、参考文献を管理している。
参考文献は`docs/source/reference.bib`に記載して、

```rst
:cite:`sod_1978JCoPh..27....1S`
```

として引用する。

GitHub Pagesで公開する場合は、`.github/workflows/deploy_pages.yml`を利用する。mainブランチにpushすると、`gh-pages`ブランチに自動的にデプロイされる。
