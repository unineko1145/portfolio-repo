from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true,y_pred,model_name="Model"):
    print(f"---{model_name}評価結果---")

    accuracy = accuracy_score(y_true,y_pred)
    print(f"精度(Accuracy):{accuracy:.4f}")

    cm =confusion_matrix(y_true,y_pred)
    print(f"\n混同行列:\n",cm)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False      
    )

    plt.title(f'{model_name}Confusion Matrix')
    plt.ylabel("Predict Label")
    plt.ylabel("True Label")
    plt.show