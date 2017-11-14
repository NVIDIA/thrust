###########################################
#
# A basic search and replace on a text file
#
###########################################

import sys

# add strings to replace here
replace_map = {'Linux': {'STDOUT thrust': 'STDOUT ../../thrust/internal/test/thrust'},
               'Windows': {'STDOUT thrust': 'STDOUT ..\\..\\thrust\\internal\\test\\thrust'}}


# searches and replaces in place, returns description and status
def search_and_replace(filename, os=None):
    if os not in replace_map:
        return "invalid os", 1

    # read all the data in the file to a string
    try:
        with open(filename, 'r') as f:
            data = f.read()
    except Exception as e:
        return "Error: {0}".format(e), 1

    # search and replace
    try:
        current_map = replace_map[os]
        for k in current_map:
            data = data.replace(k, current_map[k])
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
    if len(sys.argv) == 3:
        text, status = search_and_replace(sys.argv[1], sys.argv[2])
    else:
        text, status = "Command Format: python sar_utility <filename> <os>", 1

    print text
    sys.exit(status)


if __name__ == "__main__":
    main()
