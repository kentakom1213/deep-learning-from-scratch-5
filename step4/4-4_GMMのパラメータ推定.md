## GMMのパラメータ推定

混合ガウスモデル（GMM）は以下のような式で表される．

$$
\begin{align}
p(\boldsymbol{x}; \boldsymbol{\phi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \sum_{k = 1}^K \phi_k \mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
\end{align}
$$

ここで，データ $\mathcal{D} = \{\boldsymbol{x}^{(1)}, \boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(N)}\}$ が与えられたとき，パラメータ $\boldsymbol{\phi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}$ を最尤推定によって求めることを考える．

簡単のため，パラメータをまとめて $\boldsymbol{\theta} ~(= (\boldsymbol{\phi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}))$ とする．モデルがパラメータ $\theta$ をとるときの尤度 $p(\mathcal{D}; \boldsymbol{\theta})$ は，

$$
\begin{align}
    p(\mathcal{D}; \boldsymbol{\theta})
    &= p(\boldsymbol{x}^{(1)}; \boldsymbol{\theta}) ~ p(\boldsymbol{x}^{(2)}; \boldsymbol{\theta}) ~ \cdots ~ p(\boldsymbol{x}^{(N)}; \boldsymbol{\theta})
\end{align}
$$

と表される．

そのままでは扱いづらいので，対数をとると以下のようになる．

$$
\begin{align}
    \log p(\mathcal{D}; \boldsymbol{\theta})
    &= \log (~ p(\boldsymbol{x}^{(1)}; \boldsymbol{\theta}) ~ p(\boldsymbol{x}^{(2)}; \boldsymbol{\theta}) ~ \cdots ~ p(\boldsymbol{x}^{(N)}; \boldsymbol{\theta}) ~)\\
    &= \sum_{n = 1}^N \log p(\boldsymbol{x}^{(n)}; \boldsymbol{\theta})\\
    &= \sum_{n = 1}^N \left( \log \sum_{k = 1}^K \phi_k \mathcal{N}(\boldsymbol{x}^{(n)}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
\end{align}
$$

ここで，尤度が最も高い，すなわち $\log p(\mathcal{D}; \boldsymbol{\theta})$ を最大にするような $\theta$ が，データ $\mathcal{D}$ に最も適合したパラメータということになる，

よってこれを求めればよいが，式(5) は解析的に解くことが難しいため，別の手法が必要とされる．

---

参考：「ゼロから作るディープラーニング 5」p84
