#!/usr/bin/env python
import sys,os
import argparse
import textwrap

from networkx.utils import open_file
try:
    import cPickle as pickle
except ImportError:
    import pickle
from networkx.algorithms import isomorphism
import networkx as nx

@open_file(0, mode='rb')
def read_gpickle(path):
    return pickle.load(path)

def G_converter(file):
    ext = file.split('.')[-1]
    file = open(file,'r')
    graph = nx.Graph()
    if ext == 'mol2':
        flag = False
        for line in file:
            if line.startswith("@") and line.endswith("ATOM"):
                flag = True
                continue
            if flag:
                if line.startswith("@") and line.endswith("BOND"):
                    flag = False
                    continue
                atom_id, atom_name, x, y, z, atom_type, subst_id, subst_name = line.strip().split()[:8]
                if '.' in atom_type:
                    atom_type = atom_type.split('.')[0]
                graph.add_node(int(atom_id),**{'atomname': atom_name,
                                             'x'       : float(x),
                                             'y'       : float(y),
                                             'z'       : float(z),
                                             'element' : atom_type,
                                             'resid'   : int(subst_id),
                                             'resname' : subst_name})
        flag = False
        file.seek(0)
        for line in file:
            if line.startswith("@") and line.endswith("BOND"):
                flag = True
                continue
            if flag:
                if line.strip() == '':
                    flag = False
                    continue
                entr = line.strip().split()
                atomi = int(entr[1])
                atomj = int(entr[2])
                graph.add_edge(atomi,atomj)

    elif ext == 'mol':
        serial = 1
        flag = False
        for line in file:
            if ("V2000" or "v2000" or "V3000" or "v3000") in line:
                flag = True
                continue
            elif flag and line.startswith("M"):
                continue
            elif flag and (line[5] and line[15]) == ".":
                x, y, z, element = line[0:10], line[10:20], line[20:30], line[31:34].strip()
                graph.add_node(int(serial), **{'x'      : float(x),
                                             'y'      : float(y),
                                             'z'      : float(z),
                                             'element': element})

                serial += 1
                continue
            elif flag and len(line) == 22:
                atomi = int(line[0:3])
                atomj = int(line[3:6])
                graph.add_edge(atomi, atomj)
                continue
            elif flag and line.endswith("END"):
                flag = False
                break
    elif ext == 'sdf':
        serial = 1
        flag = False
        for line in file:
            if len(line) == 34:
                flag = True
                continue
            elif flag and (line[5] and line[15]) == ".":
                x, y, z, element = line[0:10], line[10:20], line[20:30], line[31:34].strip()
                graph.add_node(int(serial), **{'x'      : float(x),
                                             'y'      : float(y),
                                             'z'      : float(z),
                                             'element': element})

                serial += 1
                continue
            elif flag and len(line) == 19:
                atomi = int(line[0:3])
                atomj = int(line[3:6])
                graph.add_edge(atomi, atomj)
                continue
    return graph

def G_removeH(graph):
    rmH = []
    for node in graph.nodes():
        if graph.node[node]['element'] == 'H':
            rmH.append(node)
    for node in rmH:
        graph.remove_node(node)
    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pdbid', dest="pdbid")
    parser.add_argument('-resid', dest="resid")
    parser.add_argument('-csml', nargs='*', dest="csml")
    parser.add_argument('-het', nargs='*', dest="het")
    parser.add_argument('-datadir', dest="datadir")
    parser.add_argument('-localdir', dest="localdir")
    parser.add_argument('-source', dest="source")

    inputarg = parser.parse_args()
    pdbid    = inputarg.pdbid
    resid    = inputarg.resid
    csml     = inputarg.csml
    het      = inputarg.het
    datadir  = inputarg.datadir
    localdir = inputarg.localdir
    source   = inputarg.source

    archive  = datadir+"archive"
    bindir   = localdir+"bin"

    # READ ligand information and Mapping

    map_lib = {}

    str = ""

   #if (source == "CUSTOM"):
   #    str += het[0]+":\n"
   #    for i in range(0,len(csml)):
   #        str += "    "+csml[i]+": "
   #        h = nx.read_gpickle("drawing_no_h.gpickle")
   #        g = nx.read_gpickle("%s/ligandrm/no_h/total/%s" % (archive,csml[i]+".gpickle"))

   #        GM = isomorphism.GraphMatcher(h,g)
   #        GM.is_isomorphic()

   #        index = 1

   #        map_ref = GM.mapping
   #        mapping = []
   #        for j in map_ref:
   #            atomname = '"%-4s"' % g.node[map_ref[j]]["atomname"]
   #            mapping.append(atomname)
   #        map_lib[csml[i]] = mapping
   #        str += "["+', '.join(mapping)+"]\n"

   #else:
    for i in range(0,len(het)):
        str += het[i]+":\n"                                    #YAML String)HETA:
        hetmol = pdbid+"_"+het[i]+".mol"
        h = nx.Graph()
        h = G_converter(hetmol)
        if (source == "CUSTOM"):
            h = G_removeH(h)

        for j in range(0,len(csml)):
            str += "    "+csml[j]+": "                         #YAML String)    LDA:
            g = nx.read_gpickle("%s/ligandrm/no_h/total/%s" % (archive,csml[j]+".gpickle"))
            GM = isomorphism.GraphMatcher(g,h)
            if GM.is_isomorphic():
                for match in GM.isomorphisms_iter():
                    matched = True
                    for k in match:
                        if g.node[k]['element'] != h.node[match[k]]['element']:
                            matched = False
                            break
                    if matched:
                        map_ref = match
                        mapping = []
                        for l in range(0,len(map_ref)):
                            for key in map_ref:
                                if map_ref[key] == (l+1):
                                    atomname = '"%-4s"' % g.node[key]["atomname"]
                                    mapping.append(atomname)
                                    break
                        map_lib[csml[j]] = mapping
                        str += "["+', '.join(mapping)+"]\n"
                        break
            else:
                for match in GM.subgraph_isomorphisms_iter():
                    matched = True
                    for k in match:
                        if g.node[k]['element'] != h.node[match[k]]['element']:
                            matched = False
                            break
                    if matched:
                        map_ref = match
                        mapping = []
                        for l in range(0,len(map_ref)):
                            for key in map_ref:
                                if map_ref[key] == (l+1):
                                    atomname = '"%-4s"' % g.node[key]["atomname"]
                                    mapping.append(atomname)
                                    break
                        map_lib[csml[j]] = mapping
                        str += "["+', '.join(mapping)+"]\n"
                        break

    fout = open('%s_csml.yml' % resid, 'w')
    fout.write(str)
    fout.close()
