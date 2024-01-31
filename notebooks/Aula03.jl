### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ c6f1aaf8-a590-42f5-a18d-585913b24c0a
begin
    using Pkg
    pkg"activate ../."
    pkg"instantiate"
end

# ╔═╡ 67404a91-fe04-4b6f-ab9a-c6400b0d0766
pkg"add ForwardDiff"

# ╔═╡ f8c79e87-52ba-40a5-ae11-d40934161fe9
using LinearAlgebra, Plots, Random

# ╔═╡ c1c88e38-ed61-4972-ad34-6ec8e12ccb2c
md"""
# Tutorial de  Julia para Otimização
## Aula 02 - Algoritmos de Otimização em Julia

### Ministrante
- Luiz-Rafael Santos ([LABMAC/UFSC/Blumenau](http://labmac.mat.blumenau.ufsc.br))
    * Email para contato: [l.r.santos@ufsc.br](mailto:l.r.santos@ufsc.br) ou [lrsantos11@gmail.com](mailto:lrsantos11@ufsc.br)
	- Repositório do curso no [Github](https://github.com/lrsantos11/Tutorial-Julia-Opt)
"""

# ╔═╡ 7bc47d0d-ae5a-46a8-8259-d5ff97688ef5
md"""
# Introdução à otimização
O problema geral de otimização (P) pode ser dado por

$$
\begin{align}
\min f(x)\tag{P}\\ 
\text{s.a. } \ell \leq c(x) \leq u
\end{align}
$$

em que $f:D_{f}\subset \mathbb{R}^n \to \mathbb{R}$ e $c:D_{c}\subset \mathbb{R}^n\to \mathbb{R}^m$.

* $f$ é *chamada função objetivo*

* Como $c(x) = (c_1(x),\ldots,c_m(x))$, então chamamos cada $c_i$ de *restrição*

* O conjunto $\Omega := \{x\in \mathbb{R}^n\mid \ell \leq c(x) \leq u$\} é *chamado conjunto viável* ou *factível*. 
  - Caso $\Omega = \mathbb{R}^n$ dizemos que (P) *irrestrito*
  - Caso $\Omega = \emptyset$, dizemos que (P) é *inviável* ou *infactível*

* Em particular, neste tutorial:
    * $f$ e cada $c_i$ serão $C^1(\mathbb{R})$ ($C^2(\mathbb{R}$) se necessário)
    
 """

# ╔═╡ 1cf30040-5e6c-4133-a671-5f696f3b0fdf
md"""
### Minimizadores


* Diremos que $x^*\in\Omega$ é um *minimizador (global)* do problema $P$ se 

$$f(x^*) \leq f(x), \forall x\in \Omega$$

* $x^*\in\Omega$ é um *minimizador local* do problema $P$ se existe $\delta>0$ tal que

$$f(x^*) \leq f(x), \forall x\in \Omega\cap \mathcal{B}_{\delta}(x^*)$$
"""

# ╔═╡ 21fc2aab-baac-4728-910a-8532c5b0b970
md"""
### Métodos Iterativos

* Os métodos para resolver (P) são iterativos, isto é, vamos gerar computacionalmente uma sequência de pontos chamados *iterandos* $(x_k)_{k\geq 0}$ tal que a aproximação $x_{k+1}$ está bem definida quando
$$f(x_{k+1}) < f(x_k),$$
se 
$\nabla f(x_k)\neq 0$


> **Direção de descida.** Dizemos que $d\in \mathbb{R}^n$ é uma direção de descida a partir de $x$ se existe $\varepsilon>0$ tal que 
$$ f(x + \alpha d) < f(x)$$
para todo $\alpha \in (0,\varepsilon]$.

* Direções que formam ângulo menor que 90 graus com $\nabla f(x)$ são de descida
    * A direção $d = -\nabla f(x)$ é chamada *direção de máxima descida a partir de $x$*


"""

# ╔═╡ 8f68a47f-d9eb-41fc-9b7a-7be223f8be04
md"""
### Condição de otimalidade

* Queremos gerar uma suquência tal que $x_k \to x^*$ ou tal que  $x_k \to \bar x$, em que $\bar x$ satisfaz alguma *condição de otimalidade*

* Neste tutorial não discutiremos com profundidade a teoria de  *Condições de Otimalidade* mas usaremos alguns resultados como o teorema a seguir. 

> **Teorema.** [Condição necessária de 1a ordem]
>     Seja $f:\mathbb{R}^n \to \mathbb{R}$ diferenciável no ponto $x^*$. Se $x^*$ é minimizador local de $f$, então
>    $$ \nabla f(x^*) = 0.$$

* Todo $x^*$ que cumpre a condição acima é chamado *ponto estacionário* ou *ponto crítico*

* Vamos implementar métodos que convergem (em princípio) para pontos estacionários."""

# ╔═╡ ea9dcff2-1baa-41ed-b34d-c9171dd50a00
md"""
### Algoritmo básico de otimização 

* Passo 0: 
    - escolha ponto inicial $x_0$; faça $k = 0$.

* Passo 1: 
     - se  $x_k$ está perto suficiente de um *ponto estacionário*, pare; caso contrário ($\nabla f(x_k)\neq 0$), vá ao Passo 2.

* Passo 2:
	- encontre direção $d_k$ de **descida** e tamanho de passo $\alpha_k$ tal que $f(x_k + \alpha_kd_k) < f(x_k)$

* Passo 3: 
	- compute	
     $$ x_{k+1} = x_k + \alpha_k d_k$$

* Passo 4: 
    - faça k = k+1 e volte ao Passo 1
     

"""

# ╔═╡ 56818fe8-c4ac-4b63-adc2-d1697d910755
md"""
# Dois métodos para minimizar quadráticas sem restrição

* Vamos  resolver  o seguinte problema quadrático irrestrito (QP)
$$\label{eq:quadratic}
\min_{x \in \mathbb{R}^{n}} f(x)=\frac{1}{2} x^{T} A x - b^{T} x + c.
$$
com $A\in\mathbb{R}^{n\times n}$ simétrica definida postiva.

* Note que 
$$ \label{eq:gradient}
\nabla f(x) = Ax-b 
$$





* Vamos seguir de perto o texto
> *Métodos computacionais de otimização* de J. Mário Martinez e Sandra A. Santos  (1995) [(PDF)](http://www.ime.unicamp.br/~martinez/mslivro.pdf).

"""

# ╔═╡ 9bfef0bc-a076-4027-83bf-15edcb79061b
begin
    # Estrutura a e funções para a quadratica
    
    struct Quadratica
         Q::Matrix
         b::Vector
         c::Number
         Quadratica(Q,b, c) = new(Q,b,c)
    end
    
    obj(quad::Quadratica,x::Vector) = .5*dot(x,quad.Q*x) - dot(quad.b,x) + quad.c
    grad(quad::Quadratica,x::Vector) = quad.Q*x - quad.b
    

end

# ╔═╡ 7c054414-17e5-4038-9c22-6f1b3c0da40c
md"""
### Exemplo
$$
\min_{x \in \mathbb{R}^{n}} f(x)=\frac{1}{2} x^{T} Q x - b^{T} x + c
$$

$$
{Q}=\left[\begin{array}{ll}3 & 2 \\ 2 & 6\end{array}\right], {b}=\left[\begin{array}{r}2 \\ -8\end{array}\right], \quad {c}=0 \Rightarrow \mathbf{x}^*=\left[\begin{array}{r}2 \\ -2\end{array}\right]
$$"""

# ╔═╡ deb274d5-9b53-4e19-adbd-9b11d75eb94c
md"""
## Método de Máxima descida (ou gradientes descendentes ou Cauchy)"""

# ╔═╡ 888e0a1c-adee-48a8-979a-a7ebee0c97af
md"""
* No modelo geral, faremos $d_k = -\nabla f(x_k) = b-Qx$ e vamos calcular $\alpha_k$ tal que 
$$ \alpha_k = \operatorname{arg}\min_{\alpha\geq 0} \varphi(\alpha) = f(x_k + \alpha d_k)$$

* O valor de $\alpha_k$ é dado pela fórmula fechada (Por quê? - Exercício)

    $$ \alpha_k = \frac{d_k^Td_k}{d_k^TQd_k} \text{ }$$

* Direções de duas iterações consecutivas são ortogonais

* **Critério de parada**: $\| d_k\|$ pequena o suficiente"""

# ╔═╡ 3ce4a8a6-ccfc-4f94-8f35-b5a991a2b81d
md"""
### Jogo da pesca de tesouro

* Clique [aqui](https://www.i-am.ai/apps/gradient-descent/index.html) para jogar


> Após jogar algumas vezes você já percebeu que a maneira mais rápida de encontrar o ponto mais profundo no fundo do oceano é prestando atenção na inclinação encontrada em cada vez que você joga seu equipamento: o quão descendente o fundo está e em que direção estava inclindado. Embora você não possa **ver** o fundo e não tenha uma visão completa de como ele é, a inclinação te sugere por onde contiuar a busca.


"""

# ╔═╡ bc76c6d9-65e9-4bf0-bbc8-b07f5f1f88bd
md"""
## Algoritmo Gradiente Descendente


* **Passo 0.** Defina $x^0$, $d^0 =  b - Qx^0$ e $k=0$
* **Passo 1.** Enquanto $d^k \neq 0$ faça
    * $
    \alpha_{k} = \frac{({d}^{k})^Td^k}{({d}^{k})^TQ {d}^{k}}
    $
    
    * ${x}^{k+1}={x}^{k}+\alpha_{k} {d}^{k}$
    
    * $d_{k+1} = d_k - \alpha_kQd_k$ (Por quê?)
    
    * $k = k+1$
    
"""

# ╔═╡ 1df5a8ab-441f-411c-8b57-7edaef46f3f5
begin
    function iter_gradient(xₖ, dₖ,dotdₖ, quad)
        """
        Iteração basica de GD
        Parâmetros:
        xₖ: iteração atual
        dₖ: direção atual
        dotdₖ: prod interno dₖ
        quad: quadratica de interesse
        """
        Qdₖ = quad.Q*dₖ
        
        αₖ = dotdₖ / dot(dₖ,Qdₖ)
        
        xₖ = xₖ +  αₖ*dₖ
        
        dₖ = dₖ - αₖ*Qdₖ
        
        dotdₖ = dot(dₖ,dₖ)
    
        return xₖ, dₖ, dotdₖ
    end
    
    function gradient(quad::Quadratica,x₀::Vector;itmax::Int = 10, ε::Float64 = 1e-6)
    	"""
        Método de Gradientes descendentes
        Parâmetros:
        quad: Quadratica
        x₀: ponto inicial
        itmax: número max de iterçãoes de GD
        ε: tol
        """
        k = 0
        xₖ = x₀
        dₖ = - grad(quad,xₖ)
        dotdₖ = dot(dₖ,dₖ)
        X =  xₖ
        while k <= itmax && dotdₖ >= ε^2 # equivalente a norm(dₖ) <= ε
            xₖ, dₖ, dotdₖ = iter_gradient(xₖ, dₖ, dotdₖ, quad)
            X = hcat(X,xₖ)
    		k += 1 # equivale a k = k + 1    
        end
        return X, k
    end
end

# ╔═╡ 0ab2450d-ceec-4bb3-9142-3d1fa3e4616d
md"""
## Método dos Gradientes Conjugados"""

# ╔═╡ 07d2aab4-b2f3-48e9-8a4f-836c4fa0c76c
md"""
## Eliminado o erro em $n$ passos

O erro gerado por uma sequência $({x}^{k})_{k\geq 0}$ em relação a solução $x^{*}$ de (QP) é dado por
$$
{e}^{k}={x}^{k}-{x}^{*}
$$
e o resíduo é dado por
$$
{r}^{k}= - \nabla f(x^k) = {b}-{Q} {x}^{k}.
$$

Substituindo (6) in (7) obtemos
$$
\begin{aligned}
{r}^{k}&={b}-{Q}\left({e}^{k}+{x}^{*}\right)=-{Q} {x}^{*}+{b}-{Q} {e}^{k}\\ &=-{Q} {e}^{k}
\end{aligned}
$$
representando a relação entre o erro e o resíduo."""

# ╔═╡ 9293ff33-f327-4618-b833-c07964022d8c
md"""
Dado $x^{0}$, queremos definir uma sequência de $n$ direções (vetores) LI ${d}^{0}, \ldots, {d}^{n-1}$ então 
$$
\mathbb{R}^n = \mathcal{D}=\operatorname{span}\left({d}^{0}, \ldots, {d}^{n-1}\right)
$$
tais que 
$$\label{eq:xstra-x0}
{x}^{*}={x}^{0}+\sum_{i=0}^{n-1} \alpha{i} {d}^{i}. \tag{1}
$$
Vamos computar, de fato, cada passo 
$$
{x}^{k+1}={x}^{k}+\alpha_{k} {d}^{k}, \quad k=0, \ldots, n-1
$$
e como $\mathcal{D}$ é uma base de $\mathbb{R}^{n},$  vai existir a sequência de coeficientes
$$
\boldsymbol{\Phi}=\left(\alpha_{0}, \ldots, \alpha_{n-1}\right)
$$
que faz com que (1) esteja bem definido."""

# ╔═╡ ea1869b1-226d-4496-9b68-87e186f33861
md"""
Pela definição de ${e}^{0}(:=x^{0}-x^{*} )$, temos
$$
{e}^{0}=-\sum_{i=0}^{n-1} \alpha_{i} {d}^{i}.
$$
* Nosso objetivo é eliminar o erro ${e}^{{0}},$ encontrando um conjunto   $\Phi$ e tomando os passos $\alpha_{k}$ para isso. 
*  Com isto  para obter ${x}^{*}$ apenas $n$ componentes de erro devem ser eliminadas
* Isto vai significar que o algoritmo finalizará com $n$ iterações."""

# ╔═╡ 788aa9ef-28b4-4edb-b36a-a0281b6b8f1e
md"""
##  Direções Ortogonais?
* Se escolhermos ${d}^{0}, \ldots, {d}^{n-1}$ ortogonais (i.e., $({d}^{i})^T {d}^{j} = \delta_{ij})$, teríamos mais facilidade para encontrar os $\alpha_k$'s? Vejamos"""

# ╔═╡ 5b3e39fc-78c6-4cb9-89ef-61012d117482
md"""
* Daí 
    $$
    ({d}^{k})^T{e}^{0}=-\sum_{i=0}^{n-1} \alpha_{i} ({d}^{k})^T{d}^{i}=- \alpha_{k}({d}^{k})^T {d}^{k}\implies \alpha_{k}=-\frac{({d}^{k})^T{e}^{0}}{({d}^{k})^T {d}^{k}},  k=0,\ldots,n-1 
$$

    * Como
$$
{e}^{k}={e}^{0}+\sum_{i=0}^{k-1} \alpha_{i} {d}^{i}, \: k=0, \ldots, n-1
$$
também obtemos
$$
\begin{aligned}
\alpha_{k} %- \frac{({d}^{k})^{T}\left({e}^{k}-\sum_{i=0}^{k-1} \alpha_{i} {d}^{i}\right)}{({d}^{k})^{T} {d}^{k}} \\
%&=-\frac{({d}^{k})^{T} {e}^{k}-\sum_{i=0}^{k-1} \alpha_{i} ({d}^{k})^{T} {d}^{i}}{({d}^{k})^{T} {d}^{k}}\\
& = -\frac{({d}^{k})^{T} {e}^{k}}{({d}^{k})^{T} {d}^{k}}, \quad k=0, \ldots, n-1
\end{aligned}
  $$  
* **Problema:** Não conhecemos os erros $e_k$ (Por quê?)"""

# ╔═╡ 42a5f901-42af-4737-b3fe-6cd2c5558e85
md"""
## Direções $Q$-conjugadas (ou $Q$-ortogonais)

* Se escolhermos agora ${d}^{0}, \ldots, {d}^{n-1}$ *$Q$-conjugadas*, i.e.,
    $$({d}^{i})^T Q {d}^{j} = 0\text{ se } i\neq j,$$ 
     conseguiremos computar os $\alpha_k$'s
* Vejamos 
    $$
    ({d}^{k})^TQ{e}^{0}=-\sum_{i=0}^{n-1} \alpha_{i} ({d}^{k})^TQ{d}^{i}= -\alpha_{k}({d}^{k})^TQ {d}^{k}$$
    e logo
    $$\alpha_{k}=-\frac{({d}^{k})^T(Q{e}^{0})}{({d}^{k})^TQ {d}^{k}} = \frac{({d}^{k})^Tr^0}{({d}^{k})^TQ {d}^{k}},  k=0,\ldots,n-1 
$$"""

# ╔═╡ ada8e3e4-6179-450f-90d2-1dcaa2972073
md"""
* Da mesma forma como nas direções ortogonais temos
    
    $$
    \begin{aligned}
\alpha_{k} %&=- \frac{({d}^{k})^{T}Q\left({e}^{k}-\sum_{i=0}^{k-1} \alpha_{i} {d}^{i}\right)}{({d}^{k})^{T}Q {d}^{k}} \\
%&=-\frac{({d}^{k})^{T} Q{e}^{k}-\sum_{i=0}^{k-1} \alpha_{i} ({d}^{k})^{T}Q {d}^{i}}{({d}^{k})^{T} Q{d}^{k}}\\
& = -\frac{({d}^{k})^{T} Q{e}^{k}}{({d}^{k})^{T} {d}^{k}} = \frac{({d}^{k})^Tr^k}{({d}^{k})^TQ {d}^{k}}, \quad k=0, \ldots, n-1
\end{aligned}
  $$  
* Note que sempre é possível computar $r^k ( = b-Qx^k = -\nabla f(x^k))$!"""

# ╔═╡ 5c57a49b-caeb-4b83-b4dd-a2535cc01948
md"""
## Algoritmo CG
* **Passo 0.** Defina $x^0$, $d^0 = r^0 = b - Qx^0$ e $k=0$
* **Passo 1.** Enquanto $r^k = b- Qx^k \neq 0$ faça
    * $
    \alpha_{k} = \frac{({d}^{k})^Tr^k}{({d}^{k})^TQ {d}^{k}}
    $
    * ${x}^{k+1}={x}^{k}+\alpha_{k} {d}^{k}$
    
    * $k = k+1$
    """

# ╔═╡ 1eb074aa-d8a9-4481-b5be-01873584a2fe
md"""
## Como encontrar direções conjugadas $d^k$' s?
"""

# ╔═╡ 0aba6fba-7597-468e-8f51-e6b24fafc490
md"""
### Vamos usar resíduos para computar as direções conjugadas
* Atualizamos o próximo resíduo usando o resíduo anterior
$$
\begin{aligned}
{r}^{k+1} &=-{Q} {e}^{k+1} \\
&=-{Q} (x^{k+1} - x^*) \\
&=-{Q} ({x}^{k}+\alpha^{k} {d}^{k} - x^*) \\
&=-{Q}\left({e}^{k}+\alpha^{k} {d}^{k}\right) \\
&={r}^{k}-\alpha^{k} {Q} {d}^{k}, \:  k=0, \ldots, n-1
\end{aligned}
$$

* O próximo resíduo ${r}^{k+1} $ é uma combinação linear do resíduo atual ${r}^{k} $ e de $ {Q} {d}^{k}$

* Através dos ${r}^{0}, \ldots, {r}^{k}$ calcularemos as direções conjugadas ${d}^{0}, \ldots, {d}^{k}$ """

# ╔═╡ 9a428c08-5e1e-428e-a903-5356e2764ac5
md"""
## Algoritmo CG

* **Passo 0.** Defina $x^0$, $d^0 = r^0 = b - Qx^0$ e $k=0$
* **Passo 1.** Enquanto $r^k \neq 0$ faça
    * $
    \alpha_{k} = \frac{({d}^{k})^Tr^k}{({d}^{k})^TQ {d}^{k}}
    $
    * $    {x}^{k+1}={x}^{k}+\alpha_{k} {d}^{k}$
    
    * ${r}^{k+1} = {r}^{k}-\alpha^{k} {Q} {d}^{k}$
    
    * $    k = k+1$
    """

# ╔═╡ bc15f0d0-4e01-442f-b734-f1f8fb9148e2
md"""
> * **Proposição.**
>$$ \mathcal{D}_{k}:=\operatorname{span}\left\{{d}^{0}, \ldots, {d}^{k}\right\}=\operatorname{span}\left\{{r}^{0}, \ldots, {r}^{k}\right\}, \:k=0, \ldots, n-1 $$
<!--- * Como ${r}^{k+1}={r}^{k}-\alpha_{k} {A} {d}^{k}$ e além disso ${d}^{k} \in \mathcal{D}_{k}$ e ${r}^{k} \in \mathcal{D}_{k}$ então
$$
\begin{aligned}
\mathcal{D}_{k+1} &:=\operatorname{span}\left\{\mathcal{D}_{k}, {r}^{k+1}\right\} \\
&=\operatorname{span}\left\{\mathcal{D}_{k}, {r}^{k}-\alpha_{k} {A} {d}^{k}\right\} \\
&=\operatorname{span}\left\{\mathcal{D}_{k}, {A} {d}^{k}\right\}
\end{aligned}
$$ --->
"""

# ╔═╡ dd2f1ff3-32fc-4715-be0c-980cb7907e9e
md"""
> **Lema.** $r^{k+1} \perp \mathcal{D}_k = \operatorname{span}\left\{{d}^{0}, \ldots, {d}^{k}\right\}=\operatorname{span}\left\{{r}^{0}, \ldots, {r}^{k}\right\}, \:k=0, \ldots, n-1$"""

# ╔═╡ 77e9b36d-53b1-442f-b05e-827b270ce4bf
md"""
> **Teorema.** $\mathcal{D}_{k} =\operatorname{span}\left\{{r}^{0}, {Q} {r}^{0}, {Q}^{2} {r}^{0}, \ldots, {Q}^{k} {r}^{0}\right\}, \: k=0, \ldots, n-1$
     
* O subepsaçco da direita  é o *subespaço de Krylov de dimensão k+1* dado por $Q$ e $r^0$ e é denotado por  $\mathcal{K}_{k+1}(Q,r^0)$"""

# ╔═╡ ea0993ef-a515-47e8-b4d5-58a0fe8288b6
md"""
> **Corolário.** $r^{k}\perp_A\mathcal{D}_{k-2}$, para $k=0, \ldots, n-1$, isto é, $r^{k}$ é $A$-conjugado à $d^j$, $j<k-2$."""

# ╔═╡ ed463e53-42d9-4060-80f4-eeec2aeb2037
md"""
#### Consequência importante


* Se o resíduo $r^{k}$ é conjugado às direções  ${d}^{0}, \ldots, {d}^{k-2}$, então, a cada passo, por calcular o resíduo, já geramos um vetor que é  $Q$-orthogonal a todas as direções anteriores com exceção de  ${d}^{k-1}$. 

* Para obter a direção  ${d}^{k}$, que seja conjugada com as anteriores, tomamos o resíduo ${r}^{k}$ e o fazemos   $Q$-orthogonal à ${d}^{k-1}$. 

* Precisamos apenas encontrar o coeficiente  $\beta_{k-1}$ que corresponde à ${d}^{k-1}$ ao fazermos Gram-Schimidt com produto interno dado por $Q$:
$$
\begin{aligned}
{d}^{k} &={r}^{k}+\beta_{k-1} {d}^{k-1} \\
\implies 0 = ({d}^{k})^{T} {Q} {d}^{k-1} &=({r}^{k})^{T} {Q} {d}^{k-1}+\beta_{k-1} ({d}^{k-1})^{T} {Q} {d}^{k-1} \\
\implies \beta_{k-1} &=-\frac{({r}^{k})^{T} {Q} {d}^{k-1}}{({d}^{k-1})^T {Q} {d}^{k-1}}
\end{aligned}
$$"""

# ╔═╡ 14e09e94-4e37-4e80-a5f8-1031833754c6
md"""
### Forma prática do CG

* Fazendo $k = k-1$ obtemos (usando ${d}^{k} ={r}^{k}+\beta_{k-1} {d}^{k-1}$)
$$
  \alpha_{k} = \frac{({d}^{k})^Tr^k}{({d}^{k})^TQ {d}^{k}} = \frac{(r^k)^Tr^k}{({d}^{k})^TQ {d}^{k}}
$$

$$
\beta_{k}=-\frac{({r}^{k+1})^{T} {Q} {d}^{k}}{({d}^{k})^T {Q} {d}^{k}} = \frac{({r}^{k+1})^{T} {r}^{k+1}}{({r}^{k})^T {r}^{k}}
$$"""

# ╔═╡ e8c2d4d5-584a-46ef-b034-b849ccbda791
md"""
## Algoritmo CG


* **Passo 0.** Defina $x^0$, $d^0 = r^0 = b - Qx^0$ e $k=0$
* **Passo 1.** Enquanto $r^k \neq 0$ faça
    * $
    \alpha_{k} = \frac{({r}^{k})^Tr^k}{({d}^{k})^TQ {d}^{k}}
    $
    * $  {x}^{k+1}={x}^{k}+\alpha_{k} {d}^{k}$
    
    * ${r}^{k+1} = {r}^{k}-\alpha_{k} {Q} {d}^{k}$
    
    * $\beta_{k}=  \frac{({r}^{k+1})^{T} {r}^{k+1}}{({r}^{k})^T {r}^{k}}$
    
    *  ${d}^{k+1} ={r}^{k+1}+\beta_{k} {d}^{k}$
    
    * $    k = k+1$
    """

# ╔═╡ de46a2c7-64c6-43f6-a462-d3ce5755aab2
begin
    function iter_CG(xₖ, rₖ, dotrₖ,dₖ, quad, k)
        """
        Iteração basica de CG
        Parâmetros:
        xₖ: iteração atual
        rₖ: residuo atual
        dotrₖ: prod interno rₖ
        dₖ: direção atual
        quad: quadratica de interesse
        """
        Qdₖ = quad.Q*dₖ
        
    	αₖ = dotrₖ/dot(dₖ,Qdₖ)
        
        xₖ = xₖ + αₖ*dₖ
        
        if mod(k,50) != 0 
        	rₖ = rₖ - αₖ*Qdₖ
        else
        	rₖ = -grad(quad,xₖ)
        end
        
        dotrₖ_old = dotrₖ
    
        dotrₖ = dot(rₖ,rₖ)
        
        βₖ = dotrₖ/dotrₖ_old
        
        dₖ = rₖ +  βₖ*dₖ 
        
        
        return xₖ, rₖ, dotrₖ, dₖ
        
    
    end
    
    function CG(quad::Quadratica,x₀::Vector;itmax::Int = 10,ε::Float64 = 1e-8)
           """
        Método de Gradientes Conjugados
        Parâmetros:
        quad: Quadratica
        x₀: ponto inicial
        itmax: número max de iterçãoes de CG
        ε: tolerância
         """
        xₖ = x₀
        rₖ = -grad(quad,xₖ)
        dₖ = copy(rₖ)
        k = 0
        X = xₖ 
        dotrₖ = dot(rₖ,rₖ)
        while k <= itmax && dotrₖ >= ε^2
         	xₖ, rₖ, dotrₖ, dₖ = iter_CG(xₖ, rₖ, dotrₖ,dₖ, quad, k)
            X = hcat(X,xₖ)
            k += 1
        end
        return X, k
    end
end

# ╔═╡ bbe527e4-2860-41c3-a288-50db524e718e
md"""
## Exemplos com `BigFloat` e matrizes maiores"""

# ╔═╡ 9f02967d-8f44-4f8a-a1e6-cfb73540f0d2
eps()

# ╔═╡ ef6d763e-9e4c-4da9-9c66-8a8d57b82fd9
pi

# ╔═╡ 9fcf3a85-b4c0-42d8-abd1-e9860148d1c5
BigFloat(pi)

# ╔═╡ 173bc4aa-2d58-4289-be2c-4cd74e4121b2
eps(BigFloat)

# ╔═╡ 76573a3b-1a74-45b5-9657-e9673e466853
md"""
#### Mesma matrix em $2\times 2$"""

# ╔═╡ e60c9175-aa19-4dbc-9935-6cc6327c1d3b
md"""
#### Matrix de tamanho grande"""

# ╔═╡ 04ea14bf-8e98-474e-9b68-39c5f50fa252
md"""
## Métodos iterativos para minização não-linear

### E se a função não for quadrática?

* **Exemplo**  Função de [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function)

$$f(x,y) = (a-x)^2 + b(y-x^2)^2$$

- Minimizador global em $(a,a^2)$ com $f(a,a^2) = 0$
    
* Considere o modelo quadrático dado por Taylor de 2ª ordem em torno de $x_k$ para $f$

$$ m_k(d) = f(x_k) + \nabla f(x_k)^Td + \frac{1}{2}d^T\nabla^2f(x_k)d$$

* O mínimo de $m_k$, (uma quadrática), se $\nabla^2f(x_k)$ for definida positiva é a única solução do sistema linear

$$ \nabla^2f(x_k)d = -\nabla f(x_k) $$"""

# ╔═╡ 19fc8963-163b-4843-bec8-0adadef4aa71
md"""
#### Como calcular as derivadas?

* Pacote de diferenciação automática [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) para cômputo de $\nabla f(x)$ e $\nabla^2 f(x)$
"""

# ╔═╡ a6959f39-2dc9-49c8-a96d-41e95fc67678
begin
    plt = contour(x,y,quad_R2,leg=false,framestyle=:zerolines,levels=50,aspect_ratio=:equal)
    scatter!(plt,[xsol[1]],[xsol[2]])
end

# ╔═╡ e8b5bca1-8d0f-4bdd-93b7-16b2ea29b4df
md"""
## Método de Newton

### Newton para sistemas não-lineares

O método de Newton para encontrar uma solução $x^*$ do o sistema não-linear $F(x) = 0$, com $F:\mathbb{R}^n\to\mathbb{R}^n$, usa o polinômio de Taylor de 1ª ordem em torno de uma aproximação $\bar x$ obtendo o sistema linear

$$
F(\bar x) + J_F(\bar x) (x - \bar x) = F(x) = 0
$$

em que $J_F(\bar x)$ é a matriz jacobiana de $F$. 

Quando $J_F(\bar x)$ é não-singular podemos resolver o sistema linear acima pelo método iterativo

$$
x_{k+1} = x_k -  J_F(x^k)^{-1} F(x^k)
$$"""

# ╔═╡ 1f5422a6-260c-4657-96eb-4a76ed1aa4cf
md"""
### Método de Newton para Otimização

Como queremos encontrar ponto estacionário, isto é, $x^*$ tal que $\nabla f(x^*)=0$, fazendo $F = \nabla f$ obtemos o método iterativo

$$
x_{k+1} = x_k -  (\nabla^2f(x_k))^{-1} \nabla f(x_k)
$$

uma vez que $J_{\nabla f}(x) = \nabla^2f(x)$.

* A direção $d = -  (\nabla^2f(x_k))^{-1} \nabla f(x_k)$ é  exatamente a solução da minimização do modelo quadrático 

* Se $\nabla^2f(x)$ for definida positiva, a direção de Newton sempre é de descida. (Por quê?)

* Possivelmente precisamos computar um tamanho de passo $\alpha_k$ por busca linear exata (como no método de máxima descida) ou busca inexata (Passo de Armijo)
"""

# ╔═╡ c8312eac-cef2-42fa-adcb-f489e0148835
function newton(f, ∇f, H, x₀::Vector; itmax = 10_000,ε = 1e-6)
	k = 0
    xₖ = x₀
    gradₖ = ∇f(xₖ)
    while k <= itmax && norm(gradₖ) >= ε
        d = - (H(xₖ)\gradₖ)
    	xₖ = xₖ + d 
        gradₖ = ∇f(xₖ)
    	k += 1
    end
    return xₖ, k
end

# ╔═╡ 144607ef-c34e-4328-b425-41fb6a47fc92
begin
    scatter!(plt,[xsol[1]],[xsol[2]])
    scatter!(plt,[x₀[1]],[x₀[2]])
    plot!(plt,X[1,:],X[2,:],st=:path)
end

# ╔═╡ 82650b8c-3e26-4c81-a9f5-136553f01484
["LRS", pi]

# ╔═╡ ac3e49a4-fd2e-467c-911c-28b7ce962a48
begin
    x₀ = [-2.,-2]
    itmax=20
    @show X, k = CG(quad,x₀,itmax=itmax)
    plot!(plt,X[1,:],X[2,:],st=:path)
end

# ╔═╡ 1f4a22a9-178a-49e0-9087-18d6e4e6d556
begin
    quad = Quadratica(Q,b,0)
    quad_R2(x,y) = obj(quad,[x,y])
    x = range(-4,stop=6,length = 100)
    y = range(-6,stop=4,length = 100)
    z = quad_R2.(x,y)
    plt2 = surface(x,y,quad_R2,leg=false)
end

# ╔═╡ 36a8ad88-dc05-4275-b038-7a01e0786e55
begin
    x = range(-1.5, 1.5, length=400)
    y = range(-0.5 , 1.5, length=400)
    contour(x,y,(x,y) -> f([x;y]),levels=0.1:5.0:500)
    scatter!([1.0],[1.0],c=:red,label=:false)
end

# ╔═╡ 50aac538-0a70-4884-839a-b673018e8f44
begin
    Random.seed!(1234)
    N = 5000
    B = randn(N,N)
    B = 2*I + B'B
    @show cond(B)
    c = randn(N)
    x₀ = randn(N)
    quad2 = Quadratica(B,c,0)
end

# ╔═╡ 1a663f9b-d708-4526-9c60-29f18e0e74eb
begin
    quadbig = Quadratica(BigFloat.(Q),BigFloat.(b),0)
    X, k = CG(quadbig,BigFloat.(x₀))
    @show k
    X[:,end]
end

# ╔═╡ fe40f901-76fb-4529-848c-587ae2eadf1f
begin
    Q = Float64[3 2; 2 6]
    b = Float64[2, -8]
    xsol = Q\b
end

# ╔═╡ 559c0c0c-07c4-40ad-b035-79896a30c5d3
begin
    X, k = CG(quad2,x₀,itmax=5_000)
    @show k
    norm(c - quad2.Q*X[:,end])
end

# ╔═╡ ad66c429-d077-4ba0-8b3c-3688457263af
begin
    using ForwardDiff
    
    f(x) = (1-x[1])^2 + 100 * (x[2] - x[1]^2)^2
    ∇f(x) = ForwardDiff.gradient(f, x)
    H(x) = ForwardDiff.hessian(f, x)
    
    x₀ = [1.0; 1.0]
    
    mₖ(d) = f(x₀) + dot(∇f(x₀), d) + dot(H(x₀) * d, d) / 2
    q(x) = mₖ(x - x₀)
    
    a, b = 0.95,1.05
    surface(
        range(a,b, length=50),
        range(a, b, length=50),
        (x,y) -> f([x;y]),
        linealpha = 0.3,
        fc=:thermal,
        camera = (40,40))
    surface!(
        range(a, b, length=50),
        range(a, b, length=50),
        (x,y) -> q([x;y]),
    )
end

# ╔═╡ 8a95fb7e-29e5-4a10-a3e6-497f89c5ae17
x₀ = Float64[10,10]

# ╔═╡ dd7d4d7c-967b-40e4-a23c-bbc6191475ac
begin
    x₀ = [-2.,-2]
    itmax=30
    X, k = gradient(quad,x₀,itmax=itmax)
    @show norm(b - quad.Q*X[:,end])
    @show k
    X[:,end]
end

# ╔═╡ a945546d-f1fa-43da-87cb-755856cdda82
begin
    x₀ = [10,10.]
    xsol, num_iter = newton(f, ∇f, H, x₀)
end

# ╔═╡ Cell order:
# ╠═c1c88e38-ed61-4972-ad34-6ec8e12ccb2c
# ╟─7bc47d0d-ae5a-46a8-8259-d5ff97688ef5
# ╟─1cf30040-5e6c-4133-a671-5f696f3b0fdf
# ╟─21fc2aab-baac-4728-910a-8532c5b0b970
# ╟─8f68a47f-d9eb-41fc-9b7a-7be223f8be04
# ╟─ea9dcff2-1baa-41ed-b34d-c9171dd50a00
# ╟─56818fe8-c4ac-4b63-adc2-d1697d910755
# ╠═c6f1aaf8-a590-42f5-a18d-585913b24c0a
# ╠═f8c79e87-52ba-40a5-ae11-d40934161fe9
# ╠═9bfef0bc-a076-4027-83bf-15edcb79061b
# ╟─7c054414-17e5-4038-9c22-6f1b3c0da40c
# ╠═fe40f901-76fb-4529-848c-587ae2eadf1f
# ╠═1f4a22a9-178a-49e0-9087-18d6e4e6d556
# ╠═a6959f39-2dc9-49c8-a96d-41e95fc67678
# ╟─deb274d5-9b53-4e19-adbd-9b11d75eb94c
# ╟─888e0a1c-adee-48a8-979a-a7ebee0c97af
# ╟─3ce4a8a6-ccfc-4f94-8f35-b5a991a2b81d
# ╟─bc76c6d9-65e9-4bf0-bbc8-b07f5f1f88bd
# ╠═1df5a8ab-441f-411c-8b57-7edaef46f3f5
# ╠═dd7d4d7c-967b-40e4-a23c-bbc6191475ac
# ╠═144607ef-c34e-4328-b425-41fb6a47fc92
# ╟─0ab2450d-ceec-4bb3-9142-3d1fa3e4616d
# ╟─07d2aab4-b2f3-48e9-8a4f-836c4fa0c76c
# ╟─9293ff33-f327-4618-b833-c07964022d8c
# ╟─ea1869b1-226d-4496-9b68-87e186f33861
# ╟─788aa9ef-28b4-4edb-b36a-a0281b6b8f1e
# ╟─5b3e39fc-78c6-4cb9-89ef-61012d117482
# ╟─42a5f901-42af-4737-b3fe-6cd2c5558e85
# ╟─ada8e3e4-6179-450f-90d2-1dcaa2972073
# ╟─5c57a49b-caeb-4b83-b4dd-a2535cc01948
# ╟─1eb074aa-d8a9-4481-b5be-01873584a2fe
# ╟─0aba6fba-7597-468e-8f51-e6b24fafc490
# ╟─9a428c08-5e1e-428e-a903-5356e2764ac5
# ╟─bc15f0d0-4e01-442f-b734-f1f8fb9148e2
# ╟─dd2f1ff3-32fc-4715-be0c-980cb7907e9e
# ╟─77e9b36d-53b1-442f-b05e-827b270ce4bf
# ╟─ea0993ef-a515-47e8-b4d5-58a0fe8288b6
# ╟─ed463e53-42d9-4060-80f4-eeec2aeb2037
# ╟─14e09e94-4e37-4e80-a5f8-1031833754c6
# ╟─e8c2d4d5-584a-46ef-b034-b849ccbda791
# ╠═de46a2c7-64c6-43f6-a462-d3ce5755aab2
# ╠═ac3e49a4-fd2e-467c-911c-28b7ce962a48
# ╟─bbe527e4-2860-41c3-a288-50db524e718e
# ╠═9f02967d-8f44-4f8a-a1e6-cfb73540f0d2
# ╠═ef6d763e-9e4c-4da9-9c66-8a8d57b82fd9
# ╠═9fcf3a85-b4c0-42d8-abd1-e9860148d1c5
# ╠═173bc4aa-2d58-4289-be2c-4cd74e4121b2
# ╟─76573a3b-1a74-45b5-9657-e9673e466853
# ╠═1a663f9b-d708-4526-9c60-29f18e0e74eb
# ╟─e60c9175-aa19-4dbc-9935-6cc6327c1d3b
# ╠═50aac538-0a70-4884-839a-b673018e8f44
# ╠═559c0c0c-07c4-40ad-b035-79896a30c5d3
# ╟─04ea14bf-8e98-474e-9b68-39c5f50fa252
# ╟─19fc8963-163b-4843-bec8-0adadef4aa71
# ╠═67404a91-fe04-4b6f-ab9a-c6400b0d0766
# ╠═ad66c429-d077-4ba0-8b3c-3688457263af
# ╠═36a8ad88-dc05-4275-b038-7a01e0786e55
# ╟─e8b5bca1-8d0f-4bdd-93b7-16b2ea29b4df
# ╟─1f5422a6-260c-4657-96eb-4a76ed1aa4cf
# ╠═c8312eac-cef2-42fa-adcb-f489e0148835
# ╠═a945546d-f1fa-43da-87cb-755856cdda82
# ╠═8a95fb7e-29e5-4a10-a3e6-497f89c5ae17
# ╠═82650b8c-3e26-4c81-a9f5-136553f01484
