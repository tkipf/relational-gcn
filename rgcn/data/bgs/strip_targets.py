import rdflib as rdf
import gzip

g = rdf.Graph()

g.parse('./completeDataset.nt', format='nt')

lith = rdf.term.URIRef("http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis")

for s, p, o in g.triples((None, lith, None)):
    print s, p, o


g.remove((None, lith, None))

with gzip.open('bgs_stripped.nt.gz', 'wb') as output:
    g.serialize(output, format='nt')

g.close()