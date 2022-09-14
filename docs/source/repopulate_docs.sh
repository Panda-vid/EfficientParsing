source ../../venv/bin/activate
cd ../../efficient_parsing/src || exit
for dir in */; do
  sphinx-apidoc -e --implicit-namespaces -o ../../docs/source/"$dir" ./"$dir"
done