import pycurl
from io import BytesIO as BytesIO
# import xmltodict
'''not used'''


def try_normaliser(query_role):
    """Request the Stepweb normalised industry through the Stepweb API.
    Somehow the stucture of the xml returned by the API seems to change a lot
    and within a single type of structure there's plenty of variation at times."""
    print(query_role)
    prof_ind = ''
    serviceurl = "http://stepmatch-jbe-frontend.app.tjgprod.ds:4100/tools/jdnorm?query="
    urlends = "&country=uk&language=en&application=totaljobs&environment=live"
    rolename = query_role.replace(' ', '%20')
    queryurl = serviceurl + rolename + urlends
    memoryview = BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, queryurl.encode('utf-8'))
    c.setopt(c.WRITEDATA, memoryview)
    c.perform()
    c.close()
    body = memoryview.getvalue()
    str_body = body.decode('utf-8')
    results = xmltodict.parse(str_body)  # TODO needs to be rewritten with different library
    try:
        return results['NormalisationResult']['NormalisationJDs']['JD']['Discipline']['@discipline']
    except:
        try:
            for lg in range(0, len(results['NormalisationResult']['NormalisationJDs']['JD']['Discipline'])):
                if results['NormalisationResult']['NormalisationJDs']['JD']['Discipline'][lg]['@language'] == 'en':
                    prof_ind = results['NormalisationResult']['NormalisationJDs']['JD']['Discipline'][lg]['@discipline']
            z = 1
        except KeyError:
            prof_ind = None
            z = 2
        except TypeError:
            z = 3
            # for lg in range(0,len(results['NormalisationResult']['NormalisationJDs']['JD'][1]['Discipline'])):
            #    if results['NormalisationResult']['NormalisationJDs']['JD'][1]['Discipline'][lg]['@language'] == 'en':
            if 'Discipline' in results['NormalisationResult']['NormalisationJDs']['JD'][0]:
                prof_ind = results['NormalisationResult']['NormalisationJDs']['JD'][0]['Discipline'][3]['@discipline']
            else:
                # prof_ind = results['NormalisationResult']['NormalisationJDs']['JD'][1]['Discipline'][2]['@discipline']
                prof_ind = None
    return prof_ind
