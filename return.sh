#!/bin/bash 

OUTPUT='output'
INPUT='input'

mv ./$OUTPUT/*/* ./$INPUT
rm -r ./$OUTPUT/*/
