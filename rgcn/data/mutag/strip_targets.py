import rdflib as rdf
import gzip

g = rdf.Graph()

g.parse('./carcinogenesis.owl', format='xml')

is_mutagenic = rdf.term.URIRef("http://dl-learner.org/carcinogenesis#isMutagenic")

g.remove((None, is_mutagenic, None))

with gzip.open('mutag_stripped.nt.gz', 'wb') as output:
    g.serialize(output, format='nt')

g.close()
