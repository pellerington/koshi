mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE="Release" ..
make
#make install

rm ../plugin -r
mkdir ../plugin
mkdir ../plugin/resources
mkdir ../plugin/ptx
cp libHdKoshi.so ../plugin/libHdKoshi.so
cp ../plugInfo.json ../plugin/resources/plugInfo.json
find CMakeFiles/PTXKoshi.dir/src/koshi -name \*.ptx -exec cp {} ../plugin/ptx/ \;

cd ..
echo "export PXR_PLUGINPATH_NAME=$(pwd)/plugin/resources:\$PXR_PLUGINPATH_NAME" 