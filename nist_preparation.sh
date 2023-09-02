tar -xf sph2pipe_v2.5.tar.gz
cd sph2pipe_v2.5
gcc -o sph2pipe *.c -lm
cp sph2pipe ..
cd ..
rm -rf sph2pipe_v2.5
