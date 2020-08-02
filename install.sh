mkdir build
cd build/
rm include/ -r
rm lib/ -r
cmake -DCMAKE_BUILD_TYPE="Release" .. 
make install
cd ..