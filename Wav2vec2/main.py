from src.train.version.train_v1 import train_model
from src.train.predict import predict
from src.train.version.train_v2 import train_model_v2
from src.train.test.evaluate import evaluate_model
from src.train.version.train_v3 import train_model_v3
from src.train.version.train_v4 import train_model_with_validation
from src.application.app import MDDApp
import tkinter as tk
import os


if __name__ == "__main__":
    choice = int(input(
        "==================================\n"
        "[1] Train model v1.0\n"
        "[2] Train model v2.0\n"
        "[3] Train model v3.0\n"
        "[4] Train model v4.0 (add validate)\n"
        "[5] Run predict(latest_v4.0)\n"
        "[6] evaluate model(latest_v4.0)\n"
        "[7] Run demo application\n"
        "Choose: "
        ))
        
    match(choice):
        case 1:
            e = int(input(
                "\nEpoch: "
                ))
            os.system('cls' if os.name == 'nt' else 'clear')
            train_model(epoch=e)
        case 2:
            e = int(input(
                "\nEpoch: "
                ))
            os.system('cls' if os.name == 'nt' else 'clear')
            train_model_v2(epoch=e)
        case 3:
            e = int(input(
                "\nEpoch: "
                ))
            os.system('cls' if os.name == 'nt' else 'clear')
            train_model_v3(epochs=e)
        case 4:
            os.system('cls' if os.name == 'nt' else 'clear')
            train_model_with_validation()
        case 5:
            os.system('cls' if os.name == 'nt' else 'clear')
            predict()            
        case 6:
            os.system('cls' if os.name == 'nt' else 'clear')
            evaluate_model()
        case 7:
            os.system('cls' if os.name == 'nt' else 'clear')
            root = tk.Tk()
            app = MDDApp(root)
            root.mainloop()
        case _:
            print("Invalid choice")
            os.system('cls' if os.name == 'nt' else 'clear')