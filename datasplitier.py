{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from sklearn.model_selection import train_test_split\
import pandas as pd\
\
# Load dataset\
data = pd.read_csv("sleepdisorderdatabern.csv", usecols=["history", "reference", "aspects", "sentiment polarity"])\
\
# First, split 50% training and 50% remaining\
data_train, data_temp = train_test_split(data, test_size=0.5, random_state=42)\
\
# Then, split remaining 50% into 10% validation (from total) and 40% test (from total)\
validation_size = 0.2  # 10% of total is 20% of the remaining 50%\
data_val, data_test = train_test_split(data_temp, test_size=0.8, random_state=42)\
\
# Print sizes to verify\
print(f"Training set: \{len(data_train)\} samples")\
print(f"Validation set: \{len(data_val)\} samples")\
print(f"Test set: \{len(data_test)\} samples")\
}