#!/bin/bash

DATA=$1

echo -e "\033[1m ************* TEST DATA FORMAT ************* \033[0m"

bash ./test/test_all.sh ${DATA}

if [ $? -ne 0 ]; then
    echo -e "\e[31;1m ************* TEST FAILED ************* \e[0m"
    exit 1
fi
echo -e "\e[32;1m ************* TEST PASSED ************* \e[0m" 

echo -e "\033[1m ************* GENERATION BEGINS ************* \033[0m"

python generate.py --root_dir ${DATA} --version qg

if [ $? -ne 0 ]; then
    echo -e "\e[31;1m ************* GENERATION FAILED ************* \e[0m"
    exit 1
fi
echo -e "\e[32;1m ************* GENERATION DONE ************* \e[0m" 