from data.data import example, my_dataset
from modules.regression import Regression

if __name__ == '__main__':
    print(Regression.get_report(my_dataset))