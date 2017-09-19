#!/usr/bin/python3
# -*- coding: utf-8 -*- 

import os

# a = os.popen('zenity --scale --text "Escolha a velocidade da animação" --min-value=100 --max-value=1000 --value=500 --step 1').readlines()
next = os.system('zenity --question --text="É necessário mais treinamento! Proceder?"')
print(next)