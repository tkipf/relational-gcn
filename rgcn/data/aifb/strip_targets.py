import rdflib as rdf
import gzip

g = rdf.Graph()

g.parse('./aifb_fixed_complete.n3', format='n3')

employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")

rels = set(g.predicates())

g.remove((None, employs, None))
g.remove((None, affiliation, None))

with gzip.open('aifb_stripped.nt.gz', 'wb') as output:
    g.serialize(output, format='nt')

g.close()