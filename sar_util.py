###########################################
#
# A basic search and replace on a text file
#
###########################################

import sys
from operator import xor

# add strings to replace here
replace_map = {}

# searches and replaces in place, returns description and status
def search_and_replace(filename, search=None, replace=None):
    if xor(bool(search), bool(replace)):
        return "[search] [replace] should both be present", 1

    # read all the data in the file to a string
    try:
        with open(filename, 'r') as f:
            data = f.read()
    except Exception as e:
        return "Error: {0}".format(e), 1

    # search and replace
    try:
        if search and replace:
            data = data.replace(search, replace)
        else:
            for k in replace_map:
                data = data.replace(k, replace_map[k])
    except Exception as e:
        return "Error: {0}".format(e), 1

    # write new string to file
    try:
        with open(filename, 'w') as f:
            f.write(data)
    except Exception as e:
        return "Error: {0}".format(e), 1

    return "Replace successful", 0

# validates params and calls search and replace
def main():
    # validate the number of arguments
    if len(sys.argv) == 4:
        text, status = search_and_replace(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        text, status = search_and_replace(sys.argv[1])
    else:
        text, status = "Command Format: python sar_utility <filename> [search] [replace]", 1

    print text
    sys.exit(status)

if __name__ == "__main__":
    main()
