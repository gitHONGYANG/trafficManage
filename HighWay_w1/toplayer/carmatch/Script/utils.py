import os

def printlist(list, listinfo):
    print(listinfo)
    for item in list:
        print('\t', item)

def printtable(table, tableinfo):
    print(tableinfo)
    for item in table:
        print('\t', item, ' : ', table[item])

