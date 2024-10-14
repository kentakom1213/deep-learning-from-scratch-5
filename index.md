### 「ゼロから作るディープラーニング5」in Rust

---

[「ゼロから作るディープラーニング 5 （生成モデル編）」](https://www.oreilly.co.jp/books/9784814400591/)をRustで実装してみます．

---

#### 準備

##### 1. Rustのセットアップ

- https://www.rust-lang.org/ja/learn/get-started

##### 2. JupyterNotebookのセットアップ

VSCode上で実行できるようにしておくと便利です．

- https://code.visualstudio.com/docs/datascience/jupyter-notebooks

##### 3. evcxrのセットアップ

JupyterNotebook上でRustを実行できるようにしてくれます．

- https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md

---

#### その他

複数のステップで共通して利用する処理は**mymlクレート**にまとめています．

- 実装：https://github.com/kentakom1213/deep-learning-from-scratch-5/tree/main/myml
- ドキュメント：https://kentakom1213.github.io/deep-learning-from-scratch-5/myml/
