. scripts/common.sh

# for deps in emp-ot emp-tool
# do
#   if [ ! -d $BUILD_DIR/include/$deps ] 
#   then
# 	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
# 	exit 1
#   fi
# done

echo -e "build_dir:${BUILD_DIR}"
cd $BUILD_DIR/
echo -e `pwd`
cmake ..  -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=/usr/include/openssl -DCMAKE_PREFIX_PATH=$BUILD_DIR 
make
