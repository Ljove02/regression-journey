import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# SpaceX Booster Landing Prediction""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Introduction

    In this project, we developed a machine learning model to predict the landing success of the first stage of the Falcon 9 rocket, using data from historical SpaceX launches.


    The data used in this project is **not official SpaceX data**.  
    **Since SpaceX does not publicly release detailed datasets for every launch**, we worked with publicly available and community-compiled datasets.  


    Rocket reusability plays a key role in reducing the cost of space missions, which makes accurate prediction of landing outcomes highly relevant for the further development of reusable launch technologies.  

    We engineered features such as payload mass, orbit type (one-hot encoded), technical specifications of the booster (Block, Flights, Reused, GridFins, Legs), and flight number, to train a logistic regression model with polynomial expansion (degree=3).  
    The model was then evaluated using metrics such as accuracy, precision, recall, F1-score, the confusion matrix, and ROC curve, providing a detailed view of its performance.  

    Several examples of failed landings are shown below:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("/Users/veljkospasic/Desktop/Projects/Ispit ML/Ispit/images/crash.gif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Most of the failed landings were intentional and planned — SpaceX often performs controlled descents into the ocean for testing and data collection.  

    This is what the Falcon 9 rocket looks like:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("/Users/veljkospasic/Desktop/Projects/Ispit ML/Ispit/images/falcon-9-v1-2-b5__starlink-2__1.jpg")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    return go, mo, np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv(r'/Users/veljkospasic/Desktop/Projects/Ispit ML/Ispit/data/dataset_part_2.csv')
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# EDA - Exploring Data""")
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here im looking for missing values for this data set. 


    As we can see there is **26** missing values in **Landing Pad**
    """
    )
    return


@app.cell
def _(df):
    df.duplicated().any()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here im looking is there any duplicates in this dataset, and as we can see **there is none duplicates** as we got False""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Chooesing potentical features

    After examining the data, i decided to take this as main potential features.

    Now i did that based on my common knowlage about what can create biggest impact, but yet we are going to see if it was good choice.
    """
    )
    return


@app.cell
def _():
    features = [
        "FlightNumber",
        "PayloadMass",
        "Orbit",
        "Flights",
        "GridFins",
        "Reused",
        "Legs",
        "Block",
        "Class"
    ]
    return (features,)


@app.cell
def _(df, features):
    dff = df[features].copy()
    return (dff,)


@app.cell
def _(dff):
    dff
    return


@app.cell
def _(dff):
    corr_matrix = dff.drop(columns=["Orbit"]).corr()
    return (corr_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    At this point, I selected **Class** as the target variable for prediction.  
    From the earlier `data.describe()` output, we can see that the mean is around **0.66**, which indicates that about **66% of the data belongs to Class = 1** (successful landings).  
    This suggests that the dataset may be somewhat unbalanced, so it is worth exploring this further.
    """
    )
    return


@app.cell(hide_code=True)
def _(dff, plt):
    class_counts = dff["Class"].value_counts().sort_index()
    labels = ["Failed Landing (0)", "Successful Landing (1)"]
    values = [class_counts[0], class_counts[1]]
    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["red", "green"], alpha=0.7)
    plt.title("Distribution of Landing Outcomes")
    plt.ylabel("Number of Landings")
    plt.xlabel("Class")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha="center")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we can see that the data is not perfectly balanced between successful and failed landings. This is not a major issue, but it is important to keep in mind and later apply the `stratify` parameter from scikit-learn when splitting into train and test sets, to ensure an even class distribution.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Looking at out data we can see there is good amount of numeric values witch is great, but we can also see there are some colums that are bool and object.

    So we are gonna use bool one, but for now we have to ignore object since it can't be represented in correlation matrix because its object ofc.
    """
    )
    return


@app.cell(hide_code=True)
def _(corr_matrix, plt, sns):
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix (numeric + binary features)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Correlation Analysis – Key Insights

    From the correlation matrix of the selected features, we observed the following:

    - **Legs (0.67) and GridFins (0.64)** show the strongest correlation with the target variable (*Class*).  
      - This aligns with SpaceX’s landing procedures: grid fins are always deployed when a controlled landing is attempted, while landing legs are only deployed at the final stage of descent.  
      - Their presence is therefore a strong indicator of a higher probability of success, although not a guarantee (abort procedures can still occur).  
      - The typical sequence goes as follows:  
        1. Booster alignment check – if trajectory is stable, continue.  
        2. Grid fins are engaged to control descent.  
        3. At a defined distance, perform the landing burn – if trajectory remains correct, proceed.  
        4. Only at the final phase, landing legs are deployed.  
      - In practice, legs were successfully deployed about **67%** of the time, while in roughly **23%** of cases they were not, which directly reflects in the success ratio.  

    - **Block (0.42) and FlightNumber (0.40)** also show moderate correlation with *Class*.  
      - This reflects the engineering improvements of later Falcon 9 versions and the overall increased reliability of later launches.  

    - **PayloadMass (0.20), Reused (0.21), and Flights (0.15)** have weaker direct correlations with *Class*.  
      - However, they may still provide useful information when combined with other variables, especially through polynomial interactions (e.g., `PayloadMass × Block`, `Reused × Flights`).  

    ### Conclusion of this data analysis
    The strongest individual signals for predicting landing success are **Legs** and **GridFins**, followed by **Block** and **FlightNumber** as supporting factors.  
    Other features contribute less on their own but may still be informative when higher-order interactions are included (e.g., with **polynomial features of degree 3**).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Feature Engineering""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this part we are gonna:

    1. Convert features with boot to be 0 and 1 so from (True and False) -> (0, 1)
    2. We will do one-hot-encoding approach for our object data
    3. Later on we are going to use poly on deggree 3 - **NOT IN THIS SECTION**
    """
    )
    return


@app.cell(hide_code=True)
def _(dff):
    dff["GridFins"] = dff["GridFins"].map({False: 0, True: 1})
    dff["Reused"]   = dff["Reused"].map({False: 0, True: 1})
    dff["Legs"]     = dff["Legs"].map({False: 0, True: 1})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we converted bool to numeric""")
    return


@app.cell
def _(dff):
    dff
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We are checking for nan values just in case""")
    return


@app.cell(hide_code=True)
def _(dff):
    dff.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And here we are going to do one-hot-encoding useing pandas lib""")
    return


@app.cell
def _(dff, pd):
    dff_ohe = pd.get_dummies(dff, columns=["Orbit"])
    dff_f = dff_ohe.astype(int)
    return (dff_f,)


@app.cell
def _(dff_f):
    dff_f
    return


@app.cell(hide_code=True)
def _(dff_f, plt, sns):
    corr = dff_f.corr()

    plt.figure(figsize=(20,10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True)
    plt.title("Correlation Matrix – All Features")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Correlation Analysis – Extended for Orbital Insights

    From the updated correlation with our orbit data that was **OHE** in matrix we can highlight the following:

    ####**Orbit dummy variables - useing OHE:**  
      - **Negative correlations:** Orbit_GTO (-0.21), Orbit_LEO (-0.25), Orbit_ISS (-0.12 to -0.33) - missions to these orbits are less likely to result in successful landings.  
      - **Positive correlations:** Orbit_VLEO (0.17), Orbit_SSO (0.17) - missions to these orbits are more likely to result in successful landings.  
      - This matches reality where GTO and ISS missions demand higher energy, making booster recovery more challenging.  

    ### Conclusion
    The dataset is now well-prepared:  
    - We have strong direct signals (**Legs, GridFins, Block, FlightNumber**)  
    - Additional weaker signals that can be leveraged with **polynomial features**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("/Users/veljkospasic/Desktop/Projects/Ispit ML/Ispit/images/Where_and_why_we_whizz_around_Earth_pillars.jpg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Model Training""")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    import kockice as k # - moj lib
    return PolynomialFeatures, k, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this step, we split the dataset into numerical and binary features, OHE orbit features, and the target label, so that the data is properly structured for model training.

    We are going to combine it later
    """
    )
    return


@app.cell
def _(dff):
    poly_cols = ["FlightNumber", "PayloadMass", "Flights", "Block", "GridFins", "Reused", "Legs"]
    orbit_cols = [c for c in dff.columns if c.startswith("Orbit_")]
    target_col = "Class"
    return orbit_cols, poly_cols, target_col


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we split the dataset into training and testing sets using `stratified sampling` as we mentiond before, ensuring the class distribution remains balanced for model evaluation.""")
    return


@app.cell
def _(dff, orbit_cols, poly_cols, target_col, train_test_split):
    X = dff[poly_cols + orbit_cols].copy() # so here we will combine all features
    Y = dff[target_col].astype(int).copy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=42
    )

    print("Train:", X_train.shape, Y_train.shape)
    print("Test: ", X_test.shape,  Y_test.shape)
    return X_test, X_train, Y_test, Y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we take the base numerical/binary features and standardize them (mean=0, std=1), fitting on the training set and applying the same parameters to the test set.""")
    return


@app.cell
def _(X_test, X_train, k, poly_cols):
    Xtr_base = X_train[poly_cols].to_numpy().T   # (poly_cols, m_train)
    Xte_base = X_test[poly_cols].to_numpy().T    # (poly_cols, m_test)

    Xtr_std, (mu, sd) = k.standardize_rows(Xtr_base)
    Xte_std = (Xte_base - mu) / sd
    return Xte_std, Xtr_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we expand the standardized features using Polynomial Features of degree 3 to capture nonlinear relationships and interactions between variables.""")
    return


@app.cell
def _(PolynomialFeatures, Xte_std, Xtr_std):
    # Polynomial Features (degree=3)
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    Xtr_poly = poly3.fit_transform(Xtr_std.T).T
    Xte_poly = poly3.transform(Xte_std.T).T
    return Xte_poly, Xtr_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here we keep the Orbit OHE variables as 0/1 features and combine them with the polynomial features to form the final training and testing feature sets. 

    It is really important **not** to apply polynomial expansion on one-hot encoded variables, because that would only create redundant or meaningless combinations (e.g., `Orbit_GTO × Orbit_LEO`), which cannot occur at the same time and would add noise instead of useful information.
    """
    )
    return


@app.cell
def _(X_test, X_train, Xte_poly, Xtr_poly, np, orbit_cols):
    Xtr_orbit = X_train[orbit_cols].to_numpy().T
    Xte_orbit = X_test[orbit_cols].to_numpy().T

    X_train_fe = np.vstack([Xtr_poly, Xtr_orbit])
    X_test_fe  = np.vstack([Xte_poly, Xte_orbit])
    return X_test_fe, X_train_fe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we reshape the target `Class` into the form `(1, m)` so it matches the expected input/output shape for our model, and print the dimensions of both features and labels to confirm everything is aligned.""")
    return


@app.cell
def _(X_test_fe, X_train_fe, Y_test, Y_train):
    Y_trainF = Y_train.to_numpy().reshape(1, -1)
    Y_testF  = Y_test.to_numpy().reshape(1, -1)

    print("X_train_fe:", X_train_fe.shape, "| X_test_fe:", X_test_fe.shape)
    print("Y_trainF:", Y_trainF.shape, "| Y_testF:", Y_testF.shape)
    return Y_testF, Y_trainF


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This is casual marimo inputs so we can interact better with other cells, but in this one we can tweek:

    - Learning rate
    - Epochs / interations
    - L2 / regularizations
    - Threshold / if > 0.5 it landed (that by default)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # UI
    lr_slider   = mo.ui.slider(0.00, 0.20, step=0.01, label="Learning rate", value=0.02)
    ep_slider   = mo.ui.slider(5, 2000, step=5, label="Epochs", value=800)
    l2_slider   = mo.ui.slider(0.00, 0.10, step=0.001, label="L2 lambda", value=0.02)
    thr_slider  = mo.ui.slider(0.00, 1.00, step=0.01, label="Decision threshold", value=0.50)

    ui = mo.md("""
    **LR:** {lr}  
    **Epochs:** {ep}  
    **L2 λ:** {l2}  
    **Threshold:** {thr}
    """).batch(lr=lr_slider, ep=ep_slider, l2=l2_slider, thr=thr_slider)

    form = ui.form(submit_button_label="Run")
    form
    return ep_slider, form, l2_slider, lr_slider, thr_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Ignore this field this is so i can read slider values ↓""")
    return


@app.cell(hide_code=True)
def _(ep_slider, form, l2_slider, lr_slider, thr_slider):
    # Read values
    vals = form.value or {
        "lr": lr_slider.value,
        "ep": ep_slider.value,
        "l2": l2_slider.value,
        "thr": thr_slider.value,
    }

    lr         = float(vals["lr"])
    epochs     = int(vals["ep"])
    lambda_l2  = float(vals["l2"])
    threshold  = float(vals["thr"])

    lr, epochs, lambda_l2, threshold
    return epochs, lambda_l2, lr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model

    Finally, the model. As before, I reuse the library functions I implemented in the previous project, with a short recap of what each does:

    - **linear_init_params** – initializes the parameters (weights and bias).  
      *Note:* even though it says *linear*, this initialization works the same for both linear and logistic regression.  

    - **forward_logistic** – performs a forward pass (like `forward_linear`), but wrapped with the **sigmoid activation** to output probabilities.  

    - **bce_loss** – computes the **Binary Cross-Entropy (Log Loss)**, the standard loss function for logistic regression.  

    - **logistic_backprop** – computes gradients for logistic regression, taking into account the sigmoid activation.  

    - **gd_update** – performs one step of **gradient descent**, updating the parameters with the computed gradients.
    """
    )
    return


@app.cell
def _(X_train_fe, Y_trainF, epochs, k, lambda_l2, lr, np):
    params = k.init_params(n_x=X_train_fe.shape[0], n_y=1, seed=42)
    history = []

    for ep in range(epochs):
        cache = k.forward_logistic(X_train_fe, params)
        A_hat = cache["A"]                     # (1, m)
        bce = k.bce_loss(A_hat, Y_trainF)
        l2_penalty = (lambda_l2/(2*Y_trainF.shape[1])) * np.sum(params["W"]**2)
        cost = bce + l2_penalty
        grads = k.logistic_backprop(X_train_fe, Y_trainF, cache)
        grads["dW"] += (lambda_l2/Y_trainF.shape[1]) * params["W"]
        params = k.gd_update(params, grads, lr=lr)
        history.append(cost)
        if (ep+1) % 20 == 0:
            print(f"[{ep+1:4d}] BCE(train) = {cost:.6f}")

    print("Final train BCE:", history[-1])
    return history, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Graphs BCE/Epochs""")
    return


@app.cell(hide_code=True)
def _(go, history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history)+1)),
        y=history,
        mode="lines",
        name="BCE (train)"
    ))
    fig.update_layout(
        title="Logistic Regression — BCE vs Epoch",
        xaxis_title="Epoch",
        yaxis_title="BCE",
        template="plotly_white",
        width=650, height=420
    )
    fig
    return


@app.cell(hide_code=True)
def _(A_test, Y_testF, plt):
    from sklearn.metrics import roc_curve, roc_auc_score

    y_true = Y_testF.flatten().astype(int)
    y_scores = A_test

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0,1], [0,1], color="red", linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic Regression")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The ROC curve shows that the model performs well, achieving an AUC of 0.84, which indicates a strong ability to distinguish between successful and failed landings.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Accuracy""")
    return


@app.cell
def _(X_test_fe, X_train_fe, Y_testF, Y_trainF, k, params):
    cache_train = k.forward_logistic(X_train_fe, params)
    A_train = cache_train["A"].flatten()
    y_train = Y_trainF.flatten().astype(int)
    y_pred_train = (A_train >= 0.5).astype(int)
    acc_train = (y_pred_train == y_train).mean() * 100

    print(f"Train Accuracy: {acc_train:.2f}%")

    cache_test = k.forward_logistic(X_test_fe, params)
    A_test = cache_test["A"].flatten()
    y_test = Y_testF.flatten().astype(int)
    y_pred_test = (A_test >= 0.5).astype(int)
    acc_test = (y_pred_test == y_test).mean() * 100

    print(f"Test Accuracy: {acc_test:.2f}%")
    return A_test, y_pred_test, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Score - bonus

    I have to added this so i can compare my results with other on Eureka Labs
    """
    )
    return


@app.cell
def _(y_pred_test, y_test):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    prec = precision_score(y_test, y_pred_test)
    rec  = recall_score(y_test, y_pred_test)
    f1   = f1_score(y_test, y_pred_test)

    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1-score:  {f1:.2f}")

    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix:\n", cm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The model achieves balanced performance with **Precision, Recall, and F1-score all at 0.89**, which indicates that it is equally good at identifying both successful and failed landings.  
    The confusion matrix shows only a few misclassifications (2 false positives and 2 false negatives), confirming that the model generalizes well on the test set.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Future Work

    This was a fun project that took me just a few hours to put together for now, it’s the only one related to space.  
    But stay tuned: I’m already *cooking up* something much cooler! not just another notebook, but a real project with neural networks and rocket landings (hint: KSP 🚀).
    """
    )
    return


if __name__ == "__main__":
    app.run()
