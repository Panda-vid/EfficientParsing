python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install depccg
cp ./res/setup/lstm_parser_elmo.tar.gz ./venv/lib/python3.10/site-packages/depccg/models/
pip install -e .