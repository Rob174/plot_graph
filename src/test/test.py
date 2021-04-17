from src.analyser.analyse import plot_model
from src.test.models.model2 import create
import re,os

if __name__ == "__main__":
    model = create()
    path = "/".join(re.split(r'\\|/',os.path.realpath(__file__))[:-1])+"/results/result.png"
    plot_model(model,path)
