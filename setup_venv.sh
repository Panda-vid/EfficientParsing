sudo apt install gcc
sudo apt install python3.10
sudo apt install python3.10-dev default-libmysqlclient-dev build-essential libpq-dev
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install depccg
cp ./res/setup/lstm_parser_elmo.tar.gz ./venv/lib/python3.10/site-packages/depccg/models/
python -m nltk.downloader all
pip install -e .
pip install -U sphinx
pip install -U sphinx-autoapi