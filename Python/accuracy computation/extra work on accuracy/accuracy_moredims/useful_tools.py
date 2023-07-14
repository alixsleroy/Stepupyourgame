import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from settings import *
from pot_definition import *

# def dta_format_under(mat):
#     dta = pd.DataFrame(mat,columns=["sim","t","q","p","g"])
#     return dta

def dta_format_over(mat):
    dta = pd.DataFrame(mat,columns=["sim","count","t","g","gp","x","y"])
    return dta
