import csv
import collections
from clean import clean
'''used in output'''

def get_mostcommon(path, n, i=3):
    """Get the n most common rolename from csvfile (I used it) for testing purposes mostly."""
    allroles = []
    with open(path, 'rt', encoding='mac_roman') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in csvreader:
            try:
                role = clean(row[i])
                allroles.append(''.join(role))
            except IndexError:
                pass
    mostc = collections.Counter(allroles)
    roles = mostc.most_common(n)
    mostcroles = [x[0] for x in roles]
    return mostcroles
