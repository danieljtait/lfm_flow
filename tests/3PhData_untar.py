import numpy as np
from collections import namedtuple
import tarfile
import tempfile
from urllib.request import urlopen

class ThreePhaseData:

    @staticmethod
    def download():
        url = "http://staffwww.dcs.shef.ac.uk/" + \
              "people/N.Lawrence/resources/3PhData.tar.gz"

        data = {}

        with urlopen(url) as ftpstream:

            with tempfile.TemporaryFile() as tmpfile:

                while True:
                    s = ftpstream.read(16384)

                    if not s:
                        break

                    tmpfile.write(s)

                ftpstream.close()
                tmpfile.seek(0)

                tfile = tarfile.open(fileobj=tmpfile, mode="r:gz")

                for member in tfile.getmembers():
                    f = tfile.extractfile(member)
                    content = f.read().decode('utf-8')
    
                    arr = np.fromstring(content, sep=' ')

                    if member.name == 'DataTrn.txt':
                        data['DataTrn'] = arr.reshape(1000, 12)
                    elif member.name == 'DataTrnLbls.txt':
                        data['DataTrnLbls'] = arr.reshape(1000, 3)

        keys = ['DataTrn', 'DataTrnLbls']
        ThreePhData = namedtuple('ThreePhData', 'DataTrn DataTrnLbls')

        return ThreePhData(*[data[key] for key in keys])
        


