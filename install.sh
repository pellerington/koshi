mkdir build
cd build/
rm include/ -r
rm lib/ -r
mkdir include
mkdir include/koshi
rsync -q --include '*.h' --filter 'hide,! */' -avm ../src/ include/koshi/
cmake ..
make install
cd ..