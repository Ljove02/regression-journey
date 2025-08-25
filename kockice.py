import numpy as np

# Init Params for Linear Reggresion ---
def init_params(n_x: int, n_y: int = 1, seed: int | None = None):
    """
    Initialize parameters for a linear regression model.
    
    Args:
        n_x: number of input features
        n_y: output dimension
        seed: random seed for reproducibility
    
    Returns:
        params: dict with keys "W" (n_y, n_x) and "b" (n_y, 1)
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(size=(n_y, n_x))   # standard normal ~ N(0,1)
    b = np.zeros((n_y, 1))
    return {"W": W, "b": b}
# ---------------------------------------------------------------------
# Foward Pass for Linear Reggresion
def forward_linear(X: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Computes the linear forward pass: Y_hat = W X + b
    
    Args:
        X: input data of shape (n_x, m)
        parameters: dict containing:
            W -- weight matrix of shape (n_y, n_x)
            b -- bias vector of shape (n_y, 1)
    
    Returns:
        Y_hat: prediction of shape (n_y, m)
    """
    W, b = parameters["W"], parameters["b"]
    Y_hat = np.dot(W, X) + b   # (n_y, n_x) @ (n_x, m) + (n_y, 1) -> (n_y, m)
    return Y_hat
# ---------------------------------------------------------------------
# Forward Pass for Logistic Regression
def forward_logistic(X: np.ndarray, parameters: dict) -> dict:
    """
    Computes the forward pass for logistic regression: applies linear transformation
    followed by sigmoid activation.

    Args:
        X: input data of shape (n_x, m)
        parameters: dict containing:
            W -- weight matrix of shape (1, n_x)
            b -- bias vector of shape (1, 1)

    Returns:
        cache: dict with
            "Z" -- linear logits (1, m)
            "A" -- activations after sigmoid, predicted probabilities (1, m)
    """
    # Msm samo wrapujemo u sigmoid func i tjt
    Z = forward_linear(X, parameters)  # (1, m)
    A = sigmoid(Z)                     # (1, m)
    return {"Z": Z, "A": A}
# ---------------------------------------------------------------------
# MSE - Mean Square Error / Loss Function
def mse_loss(Y_hat: np.ndarray, Y: np.ndarray, use_half: bool = False) -> float:
    """
    Computes the Mean Squared Error (MSE) cost.
    
    Args:
        Y_hat: predictions of shape (n_y, m)
        Y: true labels of shape (n_y, m)
        use_half: if True, uses J = (1/(2m)) * sum((Y_hat - Y)^2) - easy for grad,
                  else J = (1/m) * sum((Y_hat - Y)^2)
    
    Returns:
        cost: scalar float (the MSE)
    """
    Y_hat = np.asarray(Y_hat, dtype=float)
    Y = np.asarray(Y, dtype=float)
    m = Y.shape[1]
    diff = Y_hat - Y
    num = np.sum(diff**2)
    return float(num/(2*m) if use_half else num/m)
# ---------------------------------------------------------------------
# Backpropagation for linear
def linear_backprop(X: np.ndarray, Y: np.ndarray, Y_hat: np.ndarray, use_half: bool = False):
    """
    Gradients for linear model (Y_hat = W X + b) with MSE loss.
    
    Args:
        X: (n_x, m)
        Y: (n_y, m)
        Y_hat: (n_y, m)
        use_half: if True -> J = (1/(2m))*sum(...), grads scale = 1/m; else 2/m

    Returns:
        grads: dict with
            "dW": (n_y, n_x)
            "db": (n_y, 1)
    """
    m = X.shape[1]
    dZ = Y_hat - Y                         # (n_y, m)
    factor = (1.0/m) if use_half else (2.0/m)
    dW = factor * (dZ @ X.T)               # (n_y, m) @ (m, n_x) -> (n_y, n_x)
    db = factor * np.sum(dZ, axis=1, keepdims=True)  # (n_y, 1)
    return {"dW": dW, "db": db}
# ---------------------------------------------------------------------
# Backpropagation for logistic
def logistic_backprop(X: np.ndarray, Y: np.ndarray, cache: dict):
    """
    Computes gradients for logistic regression with binary cross-entropy loss.

    Args:
        X: input data of shape (n_x, m)
        Y: true labels of shape (1, m), values in {0,1}
        cache: dict from forward_logistic containing:
            "Z" -- linear logits (1, m)
            "A" -- activations (1, m)

    Returns:
        grads: dict with
            "dW" -- gradient of loss with respect to W, shape (1, n_x)
            "db" -- gradient of loss with respect to b, shape (1, 1)
    """
    A = cache["A"]               # (1, m)
    m = X.shape[1]
    dZ = A - Y                   # (1, m)
    dW = (dZ @ X.T) / m          # (1, m)@(m, n_x) -> (1, n_x)
    db = np.sum(dZ, axis=1, keepdims=True) / m   # (1,1)
    return {"dW": dW, "db": db}
# ---------------------------------------------------------------------
# Gradient Descent
def gd_update(parameters: dict, grads: dict, lr: float) -> dict:
    """
    One Gradient Descent update step for linear model parameters.
    
    Args:
        parameters: {"W": (n_y, n_x), "b": (n_y, 1)}
        grads: {"dW": (n_y, n_x), "db": (n_y, 1)}
        lr: learning rate (alpha)

    Returns:
        parameters: updated in-place style (also returned for convenience)
    """
    parameters["W"] = parameters["W"] - lr * grads["dW"]
    parameters["b"] = parameters["b"] - lr * grads["db"]
    return parameters
# ---------------------------------------------------------------------
# Standard Scaler - Normalization
def standardize_rows(X: np.ndarray, eps: float = 1e-8):
    """
    Standardize features (rows) of X to mean 0 and std 1.
    Shape convention: X is (n_x, m)  -> n_x features, m samples.
    
    Args:
        X: np.ndarray of shape (n_x, m)
        eps: small constant to avoid division by zero
    
    Returns:
        Xs: standardized version of X (n_x, m)
        stats: tuple (mu, sd) for later use on test data
               mu: (n_x, 1), sd: (n_x, 1)
    """
    mu = X.mean(axis=1, keepdims=True)   # mean per feature
    sd = X.std(axis=1, keepdims=True) + eps  # std per feature
    Xs = (X - mu) / sd
    return Xs, (mu, sd)
# ---------------------------------------------------------------------
# Activation Function - Sigmoid
def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid activation function element-wise.

    Args:
        Z: input array (n_y, m)

    Returns:
        A: array of same shape as Z, values in (0,1)
    """
    # stable sigmoid
    out = np.empty_like(Z, dtype=float)
    pos = Z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    exp_z = np.exp(Z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out
# ---------------------------------------------------------------------
# Binary Cross Entrophy - Log Loss / Loss Function
def bce_loss(A: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes Binary Cross-Entropy loss for logistic regression.

    Args:
        A: predicted probabilities of shape (1, m), values in (0,1)
        Y: true labels of shape (1, m), values in {0,1}
        eps: small constant to avoid log(0)

    Returns:
        loss: scalar float, binary cross-entropy
    """
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    m = Y.shape[1]
    A = np.clip(A, eps, 1.0 - eps)
    return float(-np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / m)
# ---------------------------------------------------------------------


""" **Roadmap (kockice)**  
Optimizers (Adam), LR schedulers, early stopping → L1/L2/ElasticNet → softmax & multiclass → Dense/MLP blocks → attention (MHA, LayerNorm, residuals) → tiny Transformer encoder/decoder with checkpoints and minimal docs/examples.
"""