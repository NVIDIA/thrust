#!/bin/sh

# When a update version of CUB is fetched either from
#   http://github.com/dumerrill/PrivateCUB (currently in use)
# or
#   http://github.com/NVLabs/cub 
# Run this script from
#   //sw/gpgpu/thrust/thrust/system/cuda/detail/cub
# using the following command, only once
#  find . -type f -exec //sw/gpgpu/thrust/internal/update_cub.sh '{}' \;

# The purpose of this is to rename every instance of 
#   CUB_NSP{EFIX|OSTFIX} -> THRUST_CUB_NS_P{EFIX|OSTFIX}
# 

echo $1
cat $1|sed -e 's|CUB_NS_P|THRUST_CUB_NS_P|g' > /tmp/tmp.xxx
mv /tmp/tmp.xxx $1
