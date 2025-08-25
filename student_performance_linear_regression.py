import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Can We Predict Student Success?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Introduction

    In this project, we developed a machine learning model to predict the **Performance Index** of students based on various academic and lifestyle factors.  
    Understanding what influences student performance can provide valuable insights for improving study habits, planning educational programs, and supporting students in achieving better outcomes.  

    We used features such as hours studied, previous scores, extracurricular activities, sleep hours, and the number of practiced question papers.  
    A linear regression model with polynomial features was applied to analyze the relationship between these variables and the overall performance index.  


    The model was evaluated using metrics such as Mean Squared Error (MSE) and R² score to measure accuracy and reliability.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import kagglehub
    import plotly.graph_objects as go
    return go, kagglehub, mo, np, pd, plt, sns


@app.cell
def _(kagglehub):
    # Download latest version
    path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
    print("Path to dataset files:", path)
    return


@app.cell
def _(pd):
    df = pd.read_csv('/Users/veljkospasic/.cache/kagglehub/datasets/nikhil7280/student-performance-multiple-linear-regression/versions/1/Student_Performance.csv')
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# EDA - Data Exploration""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Exploring data type's, rows ....""")
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Looking for missing values in dataset""")
    return


@app.cell(hide_code=True)
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    df.duplicated().any()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""⚠️ Even if it says that there are duplicates, its proven from valid source there are no duplicates in data set!""")
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'No': 0, 'Yes': 1})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For "Extracurricular Activities", we converted **Yes** into int(**1**) and **No** into int(**0**)""")
    return


@app.cell(hide_code=True)
def _(df, plt):
    EA = df['Extracurricular Activities'].value_counts()


    plt.pie(EA, labels=EA.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Distribution of Students in Extracurricular Activities')
    plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see that **50.5%** students don't participate in extracurricular activities.""")
    return


@app.cell(hide_code=True)
def _(df, plt):
    sleep_hours_distribution = df["Sleep Hours"].value_counts().sort_index()
    bars = plt.bar(sleep_hours_distribution.index, sleep_hours_distribution.values)
    plt.xlabel('Sleep hours')
    plt.ylabel('Number of students')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
    plt.gca()
    return


@app.cell
def _(df):
    average_sleep_hours = df["Sleep Hours"].mean()
    average_sleep_hours
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we can see that most students sleep **8 hours**, while avrage sleep of students is **6.53 hours**""")
    return


@app.cell(hide_code=True)
def _(df, plt, sns):
    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="RdBu_r",
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title("Heatmap of Correlation Matrix")
    plt.xlabel("Features")
    plt.ylabel("Features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can see that there is a bonds between **Performance Index** and other **Features**, for example there is a great bond between **Hours Studied** and **Performance Index** of **0.37**.

    While strongest and our primary at this point is with **Previous Scores** that goes up to **0.92**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Time to train Model - Linear Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We are going to implements next:

    - Data preprocessing with Polynomial Transformation (degree = 2) - sklearn :(
    - Linear Regression
    - MSE as Loss Function
    - Gradient descent
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Chooseing features and Spliting data for model train-ing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Based on correlation matrix we saw that nearly every feature have inpact on Performance Index, with the main goal and stronges feature that is **Previous Scores**.

    Next lets split our data into **train** and **test**, im gonna use sklearn to do that.
    """
    )
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    import kockice as k # - moj lib
    return PolynomialFeatures, k, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Chooseing featues, the most important one is **Previous Scores** cuz we saw that he have the biggest correlation with **Performance **""")
    return


@app.cell
def _(df):
    feature_cols = df[["Hours Studied", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced", "Previous Scores"]]
    target_col = df[["Performance Index"]]
    return feature_cols, target_col


@app.cell
def _(feature_cols, target_col):
    X = feature_cols.to_numpy()  # (m, n_x)
    Y = target_col.to_numpy()    # (m, 1)
    return X, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now spliting data into *train* and *test* !""")
    return


@app.cell
def _(X, Y, train_test_split):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Train:", X_train.shape, Y_train.shape)
    print("Test: ", X_test.shape,  Y_test.shape)
    return X_test, X_train, Y_test, Y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Standardizeing data and Polynomial Features (Degree 2)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this section we are going to Transponse matrix of Traing and Test, then we are going to standardize data with Standard Scaler.

    Standard scaler forula is data - mean / standard_deviation
    """
    )
    return


@app.cell
def _(X_test, X_train, Y_test, Y_train, k):
    X_trainF = X_train.T   # (n_x, m_train)
    Y_trainF = Y_train.T   # (1, m_train)

    X_testF  = X_test.T    # (n_x, m_test)
    Y_testF  = Y_test.T    # (1, m_test)

    X_train_std, stats = k.standardize_rows(X_trainF)
    mu, sd = stats
    X_test_std = (X_testF - mu) / sd

    print("X_train_std:", X_train_std.shape, "| X_test_std:", X_test_std.shape)
    return X_test_std, X_train_std, Y_trainF


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we check if standard scaler did good job""")
    return


@app.cell
def _(X_train_std):
    print("Means per feature (train):", X_train_std.mean(axis=1))  # ~0
    print("Stds per feature (train):", X_train_std.std(axis=1))    # ~1
    print("First 3 samples (train, standardized):\n", X_train_std[:, :3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We are gonna use PolynomialFeatures from sklearn, because i didn't build that feature it in kocka lib""")
    return


@app.cell
def _(PolynomialFeatures, X_test_std, X_train_std):
    # degree=2
    poly2 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

    X_train_poly = poly2.fit_transform(X_train_std.T).T   # (n_x', m_train)
    X_test_poly  = poly2.transform(X_test_std.T).T        # (n_x', m_test)

    print("X_train_poly:", X_train_poly.shape, "| X_test_poly:", X_test_poly.shape)
    return X_test_poly, X_train_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Model Training""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For this this case im going to use my kockice lib that have building blocks for models.

    So syntax is maybe confusing but im gonna explain it in easy way: 

    - **linear_init_params** (initializes parameters (weights and bias), seed represents random for reproducibility)
    - **forward_linear** (typcal forward pass without activation func)
    - **mse_loss** (simle MSE here im using "cost = np.sum(diff 2) / (2 * m)" insted of "cost = np.sum(diff 2) / m" because the factor 1/2 cancels out the 2 that appears when taking the derivative of the squared term during gradient computation. )
    - **linear_backprop** ( basic backprop for forward pass without activation func)
    - **gd.update** (basic gradient descent that update all params)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""With this interactive slider's we can try and test different aproaches and see what fits the best""")
    return


@app.cell(hide_code=True)
def _(mo):
    lr_slider = mo.ui.slider(0.00, 0.20, step=0.01, label="Learning rate", value=0.03)
    ep_slider = mo.ui.slider(5, 2000, step=5, label="Epochs", value=800)
    half_check = mo.ui.checkbox(label="Use 1/m factor in loss", value=True)

    ui = mo.md("""
    **LR:** {lr}  
    **Epochs:** {ep}  
    **Use half (1/m):** {half}
    """).batch(lr=lr_slider, ep=ep_slider, half=half_check)
    form = ui.form(submit_button_label="Run")
    form 
    return ep_slider, form, half_check, lr_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is just check of values""")
    return


@app.cell(hide_code=True)
def _(ep_slider, form, half_check, lr_slider):
    vals = form.value or {
        "lr": lr_slider.value,
        "ep": ep_slider.value,
        "half": half_check.value
    }

    lr = float(vals["lr"])
    epochs = int(vals["ep"])
    use_half = bool(vals["half"])

    lr, epochs, use_half
    return epochs, lr, use_half


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And here comes the training of a simple linear regg model""")
    return


@app.cell
def _(X_train_poly, Y_trainF, epochs, k, lr, use_half):
    # epochs = 380
    # lr = 0.03
    # use_half = True  #(1/m)

    params = k.init_params(n_x=X_train_poly.shape[0], n_y=1)
    history = []

    for ep in range(epochs):
        Y_hat = k.forward_linear(X_train_poly, params)
        cost  = k.mse_loss(Y_hat, Y_trainF, use_half=use_half)
        grads = k.linear_backprop(X_train_poly, Y_trainF, Y_hat, use_half=use_half)
        params = k.gd_update(params, grads, lr=lr)
        history.append(cost)
        if (ep+1) % 20 == 0:
            print(f"[{ep+1:4d}] MSE(train) = {cost:.6f}")

    print("Final train MSE:", history[-1])
    return history, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Graphs MSE/Epochs and Typcal Linear Reggresion Graph""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### MSE/Epochs""")
    return


@app.cell(hide_code=True)
def _(history, plt):
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Linear Regression (degree=2) — MSE vs Epoch")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Typcal Linear Reggresion Graph""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This is ploty btw so u can use some tools like zoom, pan etc...""")
    return


@app.cell(hide_code=True)
def _(X_test_poly, Y_test, go, k, params):
    predict = k.forward_linear(X_test_poly, params)

    y_true = Y_test.flatten()
    y_pred = predict.flatten()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(size=6, color="blue", opacity=0.7),
        name="Predictions"
    ))

    fig.add_trace(go.Scatter(
        x=[y_true.min(), y_true.max()],
        y=[y_true.min(), y_true.max()],
        mode="lines",
        line=dict(color="red"),
        name="Ideal"
    ))

    fig.update_layout(
        title="Actual vs. Predicted Performance Index (Test Set)",
        xaxis_title="Actual Performance Index",
        yaxis_title="Predicted Performance Index",
        template="plotly_white",
        width=700,
        height=550
    )

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Accuracy""")
    return


@app.cell(hide_code=True)
def _(X_train_poly, Y_trainF, k, np, params):
    predict_train = k.forward_linear(X_train_poly, params)

    train_mse = np.mean(np.square(Y_trainF - predict_train))

    ss_res = np.sum(np.square(Y_trainF - predict_train))
    ss_tot = np.sum(np.square(Y_trainF - np.mean(Y_trainF)))
    r_squared_train = 1 - (ss_res / ss_tot)
    acc = r_squared_train * 100

    print(f"Train MSE: {train_mse:.6f}")
    print(f"Train R-squared: {r_squared_train:.6f}")
    print(f"Model Accuarcy is {acc:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Model Stat - **no PolynomialFeatures** = (165 - epoach, a = 0.2) :

    - Train MSE: 4.171156
    - Train R-squared: 0.988686
    - Model Accuarcy is 98.87%


    Model Stat - **with PolynomialFeatures** = (520 - epoach, a = 0.2) :

    - Train MSE: 4.179392
    - Train R-squared: 0.988664
    - Model Accuarcy is 98.87%
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conclusion

    The results show that both the linear model and the polynomial model (degree=2) achieved almost identical performance.  
    Without polynomial features, the model reached **MSE = 4.17**, **R² = 0.9887**, and **Accuracy = 98.87%**.  
    With polynomial features, the metrics remained essentially the same (**MSE = 4.18**, **R² = 0.9887**, **Accuracy = 98.87%**).  

    This indicates that the dataset has a predominantly linear structure, and adding polynomial terms did not significantly improve the model’s ability to explain variance or increase prediction accuracy.  
    The linear regression model alone is therefore sufficient to capture the relationship between the features and the performance index.
    """
    )
    return


if __name__ == "__main__":
    app.run()
