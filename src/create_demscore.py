#create demscore column in all SP files in nhats.db sqlite file

import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
import math
import random
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#connect to nhats.db
conn = sqlite3.connect('nhats.db')
c = conn.cursor()

#read in all SP files out of 12
for i in range(1, 13):
    c.execute("ALTER TABLE NHATS_Round_" + str(i) + "_SP_File ADD COLUMN predicted_demclas INTEGER")


