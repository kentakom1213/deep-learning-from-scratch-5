### 「ゼロから作るディープラーニング 5」in Rust

---

#### このサイトについて

このサイトは[『ゼロから作るディープラーニング 5 （生成モデル編）』（オライリー・ジャパン）](https://www.oreilly.co.jp/books/9784814400591/)の内容を Rust で実装した際のメモをまとめたものです．

記事中の文章や数式は上記の書籍を参考にしたものであり，
その他の参考文献や引用がある場合は該当する箇所で出典を示します．

また，このサイトのコードは基本的に[サンプルコード](https://github.com/oreilly-japan/deep-learning-from-scratch-5/tree/main)（MIT ライセンス）を Rust で書き直したものであり，本記事のコードについても[MIT ライセンス](https://opensource.org/license/MIT)を付与するものとします．

---

#### 準備

##### 1. Rust のセットアップ

- https://www.rust-lang.org/ja/learn/get-started

##### 2. JupyterNotebook のセットアップ

VSCode 上で実行できるようにしておくと便利です．

- https://code.visualstudio.com/docs/datascience/jupyter-notebooks

##### 3. evcxr のセットアップ

JupyterNotebook 上で Rust を実行できるようにしてくれます．

- https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md

---

#### その他

複数のステップで共通して利用する処理は**myml クレート**にまとめています．

- 実装：https://github.com/kentakom1213/deep-learning-from-scratch-5/tree/main/myml
- ドキュメント：https://kentakom1213.github.io/deep-learning-from-scratch-5/myml/
