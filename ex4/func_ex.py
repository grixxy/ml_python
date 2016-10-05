#!/usr/bin/python

def lite(a,b,c):
    return "%r,%r,%r" % (a,b,c)

def big(func): # func = callable()
    print (func())

def big2(func): # func = callable with one argument
    print (func("anything"))


def main():
    param1 = 1
    param2 = 2
    param3 = 3

    big2(lambda x: lite(param1, param2, param3))

    def lite_with_params():
        return lite(param1,param2,param3)

    big(lite_with_params)

main()