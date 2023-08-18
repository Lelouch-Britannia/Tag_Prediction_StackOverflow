import os
import subprocess

def run_model_script(filename):
    try:
        result = subprocess.run(['python', filename], capture_output=True, text=True)
        print(f"Running {filename}...\n")
        print(result.stdout)
    except Exception as e:
        print(f"Error running {filename}: {e}")

def main():
    model_files = [
        "Decision_Tree.py",
        "ExtraTreesClassifier.py",
        "KNeighborsClassifier.py",
        "Logistic Regression.py",
        "MLPClassifier.py",
        "NaiveBayes.py",
        "Neural Network_2.py",
        "Neural Network.py",
        "RandomForest.py",
        "RidgeClassifierCV.py",
        "Support_Vector_Classifier.py"
    ]

    for file in model_files:
        if os.path.exists(file):
            run_model_script(file)
        else:
            print(f"{file} does not exist. Skipping...\n")

if __name__ == "__main__":
    main()
