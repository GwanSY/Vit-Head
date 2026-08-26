. scripts/common.sh

check_tools

# 这个条件语句检查当前目录是否存在 .git 目录
# .git 目录是 Git 仓库的标志，表示当前目录是一个 Git 仓库
if [ -d .git ]; then 
  git submodule init
  git submodule update
else
  git clone https://github.com/emp-toolkit/emp-tool.git $DEPS_DIR/emp-tool
  git clone https://github.com/emp-toolkit/emp-ot.git $DEPS_DIR/emp-ot
  git clone https://github.com/libigl/eigen.git $DEPS_DIR/eigen
  # git clone https://github.com/facebook/zstd.git $DEPS_DIR/zstd
  # git clone https://github.com/intel/hexl.git $DEPS_DIR/hexl
  # git clone https://github.com/microsoft/SEAL.git $DEPS_DIR/SEAL
fi
echo $WORK_DIR
echo $BUILD_DIR
echo $DEPS_DIR

target=emp-tool
cd $DEPS_DIR/$target
git checkout 44b1dde
patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/emp-tool.patch -d $DEPS_DIR/$target
mkdir -p $BUILD_DIR/deps/$target
cd $BUILD_DIR/deps/$target
cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR
make install -j2

target=emp-ot
cd $DEPS_DIR/$target
git checkout 7f3d4f0
mkdir -p $BUILD_DIR/deps/$target
cd $BUILD_DIR/deps/$target
cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DCMAKE_PREFIX_PATH=$BUILD_DIR
make install -j2

target=eigen
cd $DEPS_DIR/$target
git checkout 1f05f51 #v3.3.3
mkdir -p $BUILD_DIR/deps/$target
cd $BUILD_DIR/deps/$target
cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR
make install -j2


for deps in emp-ot emp-tool
do
  if [ ! -d $BUILD_DIR/include/$deps ] 
  then
	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
	exit 1
  fi
done