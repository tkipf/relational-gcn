import rdflib as rdf
import gzip

g = rdf.Graph()

with gzip.open('am-combined.nt.gz', 'rb') as input:
    g.parse(input, format='nt')

rel = rdf.term.URIRef("http://purl.org/collections/nl/am/objectCategory")
g.remove((None, rel, None))

rel = rdf.term.URIRef("http://purl.org/collections/nl/am/material")
g.remove((None, rel, None))

with gzip.open('am_stripped.nt.gz', 'wb') as output:
    g.serialize(output, format='nt')

g.close()