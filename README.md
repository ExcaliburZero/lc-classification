# lc-classification
Attempts at classifying light curves.

## Setup
```
virtualenv -p /usr/bin/python3.5 env --no-site-packages
source env/bin/activate env
pip install -r requirements.txt
cd lcclassification
wget http://www.cs.oswego.edu/~cwells2/documents/curves.tar.gz
wget http://www.cs.oswego.edu/~cwells2/documents/CRTS_Varcat.csv
tar -zxvf curves.tar.gz
python cnn_01.py
```
