#include <torch/extension.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <bit>
#include <pybind11/stl_bind.h>

using namespace std;

unsigned NUM_COORDS_PER_RES = 14;
unsigned SC_ANGLES_START_POS = 6;
unsigned NUM_ANGLES = 12;


struct BackboneBuildInfo {
    const map<string, float> BONDLENS = { 
            {"n-ca", 1.442},
            {"ca-c", 1.498},
            {"c-n", 1.379},
            {"c-o", 1.229},
            {"c-oh", 1.364}
        };
    const map<string, float> BONDANGS = {
            {"ca-c-o", 2.0944}, 
            {"ca-c-oh", 2.0944}
        };
        
    const map<string, float> BONDTORSIONS = {
            {"n-ca-c-n", -0.785398163}
        };
        
    const torch::Tensor ALL_BB_LENS = torch::tensor({BONDLENS.at("c-n"),BONDLENS.at("n-ca"),BONDLENS.at("ca-c")});
};

BackboneBuildInfo BB_BUILD_INFO;

struct SCResidueInfo {
    vector<string> angles_names;
    vector<string> angles_types;
    vector<float> angles_vals;
    vector<string> atom_names;
    vector<string> bonds_names;
    vector<string> bonds_types;
    vector<float> bonds_vals;
    vector<string> torsion_names;
    vector<string> torsion_types;
    vector<string> torsion_vals;
};

map<string, SCResidueInfo> SC_BUILD_INFO = {
    {"ALA", {
        { "N-CA-CB" },
        { "N -CX-CT" },
        { 1.91463 },
        { "CB" },
        { "CA-CB" },
        { "CX-CT" },
        { 1.526 },
        { "C-N-CA-CB" },
        { "C -N -CX-CT" },
        { "p" },
    }},
    {"ARG", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD", "CG-CD-NE", "CD-NE-CZ", "NE-CZ-NH1", "NE-CZ-NH2" },
        { "N -CX-C8", "CX-C8-C8", "C8-C8-C8", "C8-C8-N2", "C8-N2-CA", "N2-CA-N2", "N2-CA-N2" },
        { 1.91463, 1.91114, 1.91114, 1.94081, 2.15025, 2.0944, 2.0944 },
        { "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2" },
        { "CA-CB", "CB-CG", "CG-CD", "CD-NE", "NE-CZ", "CZ-NH1", "CZ-NH2" },
        { "CX-C8", "C8-C8", "C8-C8", "C8-N2", "N2-CA", "CA-N2", "CA-N2" },
        { 1.526, 1.526, 1.526, 1.463, 1.34, 1.34, 1.34 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD", "CB-CG-CD-NE", "CG-CD-NE-CZ", "CD-NE-CZ-NH1", "CD-NE-CZ-NH2" },
        { "C -N -CX-C8", "N -CX-C8-C8", "CX-C8-C8-C8", "C8-C8-C8-N2", "C8-C8-N2-CA", "C8-N2-CA-N2", "C8-N2-CA-N2" },
        { "p", "p", "p", "p", "p", "p", "i" },
    }},
    {"ASN", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-OD1", "CB-CG-ND2" },
        { "N -CX-2C", "CX-2C-C ", "2C-C -O ", "2C-C -N " },
        { 1.91463, 1.93906, 2.10138, 2.03505 },
        { "CB", "CG", "OD1", "ND2" },
        { "CA-CB", "CB-CG", "CG-OD1", "CG-ND2" },
        { "CX-2C", "2C-C ", "C -O ", "C -N " },
        { 1.526, 1.522, 1.229, 1.335 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-OD1", "CA-CB-CG-ND2" },
        { "C -N -CX-2C", "N -CX-2C-C ", "CX-2C-C -O ", "CX-2C-C -N " },
        { "p", "p", "p", "i" },
    }},
    {"ASP", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-OD1", "CB-CG-OD2" },
        { "N -CX-2C", "CX-2C-CO", "2C-CO-O2", "2C-CO-O2" },
        { 1.91463, 1.93906, 2.04204, 2.04204 },
        { "CB", "CG", "OD1", "OD2" },
        { "CA-CB", "CB-CG", "CG-OD1", "CG-OD2" },
        { "CX-2C", "2C-CO", "CO-O2", "CO-O2" },
        { 1.526, 1.522, 1.25, 1.25 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-OD1", "CA-CB-CG-OD2" },
        { "C -N -CX-2C", "N -CX-2C-CO", "CX-2C-CO-O2", "CX-2C-CO-O2" },
        { "p", "p", "p", "i" },
    }},
    {"CYS", {
        { "N-CA-CB", "CA-CB-SG" },
        { "N -CX-2C", "CX-2C-SH" },
        { 1.91463, 1.89543 },
        { "CB", "SG" },
        { "CA-CB", "CB-SG" },
        { "CX-2C", "2C-SH" },
        { 1.526, 1.81 },
        { "C-N-CA-CB", "N-CA-CB-SG" },
        { "C -N -CX-2C", "N -CX-2C-SH" },
        { "p", "p" },
    }},
    {"GLN", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD", "CG-CD-OE1", "CG-CD-NE2" },
        { "N -CX-2C", "CX-2C-2C", "2C-2C-C ", "2C-C -O ", "2C-C -N " },
        { 1.91463, 1.91114, 1.93906, 2.10138, 2.03505 },
        { "CB", "CG", "CD", "OE1", "NE2" },
        { "CA-CB", "CB-CG", "CG-CD", "CD-OE1", "CD-NE2" },
        { "CX-2C", "2C-2C", "2C-C ", "C -O ", "C -N " },
        { 1.526, 1.526, 1.522, 1.229, 1.335 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD", "CB-CG-CD-OE1", "CB-CG-CD-NE2" },
        { "C -N -CX-2C", "N -CX-2C-2C", "CX-2C-2C-C ", "2C-2C-C -O ", "2C-2C-C -N " },
        { "p", "p", "p", "p", "i" },
    }},
    {"GLU", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD", "CG-CD-OE1", "CG-CD-OE2" },
        { "N -CX-2C", "CX-2C-2C", "2C-2C-CO", "2C-CO-O2", "2C-CO-O2" },
        { 1.91463, 1.91114, 1.93906, 2.04204, 2.04204 },
        { "CB", "CG", "CD", "OE1", "OE2" },
        { "CA-CB", "CB-CG", "CG-CD", "CD-OE1", "CD-OE2" },
        { "CX-2C", "2C-2C", "2C-CO", "CO-O2", "CO-O2" },
        { 1.526, 1.526, 1.522, 1.25, 1.25 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD", "CB-CG-CD-OE1", "CB-CG-CD-OE2" },
        { "C -N -CX-2C", "N -CX-2C-2C", "CX-2C-2C-CO", "2C-2C-CO-O2", "2C-2C-CO-O2" },
        { "p", "p", "p", "p", "i" },
    }},
    {"GLY", {
        {  },
        {  },
        {  },
        {  },
        {  },
        {  },
        {  },
        {  },
        {  },
        {  },
    }},
    {"HIS", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-ND1", "CG-ND1-CE1", "ND1-CE1-NE2", "CE1-NE2-CD2" },
        { "N -CX-CT", "CX-CT-CC", "CT-CC-NA", "CC-NA-CR", "NA-CR-NB", "CR-NB-CV" },
        { 1.91463, 1.97397, 2.0944, 1.88496, 1.88496, 1.88496 },
        { "CB", "CG", "ND1", "CE1", "NE2", "CD2" },
        { "CA-CB", "CB-CG", "CG-ND1", "ND1-CE1", "CE1-NE2", "NE2-CD2" },
        { "CX-CT", "CT-CC", "CC-NA", "NA-CR", "CR-NB", "NB-CV" },
        { 1.526, 1.504, 1.385, 1.343, 1.335, 1.394 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-ND1", "CB-CG-ND1-CE1", "CG-ND1-CE1-NE2", "ND1-CE1-NE2-CD2" },
        { "C -N -CX-CT", "N -CX-CT-CC", "CX-CT-CC-NA", "CT-CC-NA-CR", "CC-NA-CR-NB", "NA-CR-NB-CV" },
        { "p", "p", "p", "3.141592653589793", "0.0", "0.0" },
    }},
    {"ILE", {
        { "N-CA-CB", "CA-CB-CG1", "CB-CG1-CD1", "CA-CB-CG2" },
        { "N -CX-3C", "CX-3C-2C", "3C-2C-CT", "CX-3C-CT" },
        { 1.91463, 1.91114, 1.91114, 1.91114 },
        { "CB", "CG1", "CD1", "CG2" },
        { "CA-CB", "CB-CG1", "CG1-CD1", "CB-CG2" },
        { "CX-3C", "3C-2C", "2C-CT", "3C-CT" },
        { 1.526, 1.526, 1.526, 1.526 },
        { "C-N-CA-CB", "N-CA-CB-CG1", "CA-CB-CG1-CD1", "N-CA-CB-CG2" },
        { "C -N -CX-3C", "N -CX-3C-2C", "CX-3C-2C-CT", "N -CX-3C-CT" },
        { "p", "p", "p", "p" },
    }},
    {"LEU", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD1", "CB-CG-CD2" },
        { "N -CX-2C", "CX-2C-3C", "2C-3C-CT", "2C-3C-CT" },
        { 1.91463, 1.91114, 1.91114, 1.91114 },
        { "CB", "CG", "CD1", "CD2" },
        { "CA-CB", "CB-CG", "CG-CD1", "CG-CD2" },
        { "CX-2C", "2C-3C", "3C-CT", "3C-CT" },
        { 1.526, 1.526, 1.526, 1.526 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD1", "CA-CB-CG-CD2" },
        { "C -N -CX-2C", "N -CX-2C-3C", "CX-2C-3C-CT", "CX-2C-3C-CT" },
        { "p", "p", "p", "p" },
    }},
    {"LYS", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD", "CG-CD-CE", "CD-CE-NZ" },
        { "N -CX-C8", "CX-C8-C8", "C8-C8-C8", "C8-C8-C8", "C8-C8-N3" },
        { 1.91463, 1.91114, 1.91114, 1.91114, 1.94081 },
        { "CB", "CG", "CD", "CE", "NZ" },
        { "CA-CB", "CB-CG", "CG-CD", "CD-CE", "CE-NZ" },
        { "CX-C8", "C8-C8", "C8-C8", "C8-C8", "C8-N3" },
        { 1.526, 1.526, 1.526, 1.526, 1.471 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD", "CB-CG-CD-CE", "CG-CD-CE-NZ" },
        { "C -N -CX-C8", "N -CX-C8-C8", "CX-C8-C8-C8", "C8-C8-C8-C8", "C8-C8-C8-N3" },
        { "p", "p", "p", "p", "p" },
    }},
    {"MET", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-SD", "CG-SD-CE" },
        { "N -CX-2C", "CX-2C-2C", "2C-2C-S ", "2C-S -CT" },
        { 1.91463, 1.91114, 2.00189, 1.72613 },
        { "CB", "CG", "SD", "CE" },
        { "CA-CB", "CB-CG", "CG-SD", "SD-CE" },
        { "CX-2C", "2C-2C", "2C-S ", "S -CT" },
        { 1.526, 1.526, 1.81, 1.81 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-SD", "CB-CG-SD-CE" },
        { "C -N -CX-2C", "N -CX-2C-2C", "CX-2C-2C-S ", "2C-2C-S -CT" },
        { "p", "p", "p", "p" },
    }},
    {"PHE", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD1", "CG-CD1-CE1", "CD1-CE1-CZ", "CE1-CZ-CE2", "CZ-CE2-CD2" },
        { "N -CX-CT", "CX-CT-CA", "CT-CA-CA", "CA-CA-CA", "CA-CA-CA", "CA-CA-CA", "CA-CA-CA" },
        { 1.91463, 1.98968, 2.0944, 2.0944, 2.0944, 2.0944, 2.0944 },
        { "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2" },
        { "CA-CB", "CB-CG", "CG-CD1", "CD1-CE1", "CE1-CZ", "CZ-CE2", "CE2-CD2" },
        { "CX-CT", "CT-CA", "CA-CA", "CA-CA", "CA-CA", "CA-CA", "CA-CA" },
        { 1.526, 1.51, 1.4, 1.4, 1.4, 1.4, 1.4 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD1", "CB-CG-CD1-CE1", "CG-CD1-CE1-CZ", "CD1-CE1-CZ-CE2", "CE1-CZ-CE2-CD2" },
        { "C -N -CX-CT", "N -CX-CT-CA", "CX-CT-CA-CA", "CT-CA-CA-CA", "CA-CA-CA-CA", "CA-CA-CA-CA", "CA-CA-CA-CA" },
        { "p", "p", "p", "3.141592653589793", "0.0", "0.0", "0.0" },
    }},
    {"PRO", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD" },
        { "N -CX-CT", "CX-CT-CT", "CT-CT-CT" },
        { 1.91463, 1.91114, 1.91114 },
        { "CB", "CG", "CD" },
        { "CA-CB", "CB-CG", "CG-CD" },
        { "CX-CT", "CT-CT", "CT-CT" },
        { 1.526, 1.526, 1.526 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD" },
        { "C -N -CX-CT", "N -CX-CT-CT", "CX-CT-CT-CT" },
        { "p", "p", "p" },
    }},
    {"SER", {
        { "N-CA-CB", "CA-CB-OG" },
        { "N -CX-2C", "CX-2C-OH" },
        { 1.91463, 1.91114 },
        { "CB", "OG" },
        { "CA-CB", "CB-OG" },
        { "CX-2C", "2C-OH" },
        { 1.526, 1.41 },
        { "C-N-CA-CB", "N-CA-CB-OG" },
        { "C -N -CX-2C", "N -CX-2C-OH" },
        { "p", "p" },
    }},
    {"THR", {
        { "N-CA-CB", "CA-CB-OG1", "CA-CB-CG2" },
        { "N -CX-3C", "CX-3C-OH", "CX-3C-CT" },
        { 1.91463, 1.91114, 1.91114 },
        { "CB", "OG1", "CG2" },
        { "CA-CB", "CB-OG1", "CB-CG2" },
        { "CX-3C", "3C-OH", "3C-CT" },
        { 1.526, 1.41, 1.526 },
        { "C-N-CA-CB", "N-CA-CB-OG1", "N-CA-CB-CG2" },
        { "C -N -CX-3C", "N -CX-3C-OH", "N -CX-3C-CT" },
        { "p", "p", "p" },
    }},
    {"TRP", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD1", "CG-CD1-NE1", "CD1-NE1-CE2", "NE1-CE2-CZ2", "CE2-CZ2-CH2", "CZ2-CH2-CZ3", "CH2-CZ3-CE3", "CZ3-CE3-CD2" },
        { "N -CX-CT", "CX-CT-C*", "CT-C*-CW", "C*-CW-NA", "CW-NA-CN", "NA-CN-CA", "CN-CA-CA", "CA-CA-CA", "CA-CA-CA", "CA-CA-CB" },
        { 1.91463, 2.0176, 2.18166, 1.89717, 1.94779, 2.3178, 2.0944, 2.0944, 2.0944, 2.0944 },
        { "CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2" },
        { "CA-CB", "CB-CG", "CG-CD1", "CD1-NE1", "NE1-CE2", "CE2-CZ2", "CZ2-CH2", "CH2-CZ3", "CZ3-CE3", "CE3-CD2" },
        { "CX-CT", "CT-C*", "C*-CW", "CW-NA", "NA-CN", "CN-CA", "CA-CA", "CA-CA", "CA-CA", "CA-CB" },
        { 1.526, 1.495, 1.352, 1.381, 1.38, 1.4, 1.4, 1.4, 1.4, 1.404 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD1", "CB-CG-CD1-NE1", "CG-CD1-NE1-CE2", "CD1-NE1-CE2-CZ2", "NE1-CE2-CZ2-CH2", "CE2-CZ2-CH2-CZ3", "CZ2-CH2-CZ3-CE3", "CH2-CZ3-CE3-CD2" },
        { "C -N -CX-CT", "N -CX-CT-C*", "CX-CT-C*-CW", "CT-C*-CW-NA", "C*-CW-NA-CN", "CW-NA-CN-CA", "NA-CN-CA-CA", "CN-CA-CA-CA", "CA-CA-CA-CA", "CA-CA-CA-CB" },
        { "p", "p", "p", "3.141592653589793", "0.0", "3.141592653589793", "3.141592653589793", "0.0", "0.0", "0.0" },
    }},
    {"TYR", {
        { "N-CA-CB", "CA-CB-CG", "CB-CG-CD1", "CG-CD1-CE1", "CD1-CE1-CZ", "CE1-CZ-OH", "CE1-CZ-CE2", "CZ-CE2-CD2" },
        { "N -CX-CT", "CX-CT-CA", "CT-CA-CA", "CA-CA-CA", "CA-CA-C ", "CA-C -OH", "CA-C -CA", "C -CA-CA" },
        { 1.91463, 1.98968, 2.0944, 2.0944, 2.0944, 2.0944, 2.0944, 2.0944 },
        { "CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2" },
        { "CA-CB", "CB-CG", "CG-CD1", "CD1-CE1", "CE1-CZ", "CZ-OH", "CZ-CE2", "CE2-CD2" },
        { "CX-CT", "CT-CA", "CA-CA", "CA-CA", "CA-C ", "C -OH", "C -CA", "CA-CA" },
        { 1.526, 1.51, 1.4, 1.4, 1.409, 1.364, 1.409, 1.4 },
        { "C-N-CA-CB", "N-CA-CB-CG", "CA-CB-CG-CD1", "CB-CG-CD1-CE1", "CG-CD1-CE1-CZ", "CD1-CE1-CZ-OH", "CD1-CE1-CZ-CE2", "CE1-CZ-CE2-CD2" },
        { "C -N -CX-CT", "N -CX-CT-CA", "CX-CT-CA-CA", "CT-CA-CA-CA", "CA-CA-CA-C ", "CA-CA-C -OH", "CA-CA-C -CA", "CA-C -CA-CA" },
        { "p", "p", "p", "3.141592653589793", "0.0", "3.141592653589793", "0.0", "0.0" },
    }},
    {"VAL", {
        { "N-CA-CB", "CA-CB-CG1", "CA-CB-CG2" },
        { "N -CX-3C", "CX-3C-CT", "CX-3C-CT" },
        { 1.91463, 1.91114, 1.91114 },
        { "CB", "CG1", "CG2" },
        { "CA-CB", "CB-CG1", "CB-CG2" },
        { "CX-3C", "3C-CT", "3C-CT" },
        { 1.526, 1.526, 1.526 },
        { "C-N-CA-CB", "N-CA-CB-CG1", "N-CA-CB-CG2" },
        { "C -N -CX-3C", "N -CX-3C-CT", "N -CX-3C-CT" },
        { "p", "p", "p" },
    }}
};

vector<char> aa = {'A','C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'};
vector<string> aa3 = {"ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU","MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"};

struct AA2I: public map<char,int> {
    
    AA2I() {
        for(unsigned i = 0, n = aa.size(); i < n; i++) {
            (*this)[aa[i]] = i;
        }
    };
        
} aa2i;

struct TensorBuildInfo {
    torch::Tensor sc_source_atom;
    torch::Tensor sc_bond_length;
    torch::Tensor sc_ctheta;
    torch::Tensor sc_stheta;
    torch::Tensor sc_cchi;
    torch::Tensor sc_schi;
    torch::Tensor sc_type;  // 0: no atom, 1: regular torsion, 2: offset torsion, 3: constant
                            //index of chi is same as atom index, unless offset torsion in which case it is one before

    TensorBuildInfo() {
        unsigned N = NUM_COORDS_PER_RES-4;
        sc_source_atom = torch::zeros({20,N});
        sc_bond_length =  torch::zeros({20,N});
        sc_ctheta = torch::zeros({20,N});
        sc_stheta =  torch::zeros({20,N});
        sc_cchi =  torch::zeros({20,N});
        sc_schi =  torch::zeros({20,N});
        sc_type =  torch::zeros({20,N});
        for(unsigned a = 0; a < 20; a++) {
            string a3 = aa3[a];
            const auto& info = SC_BUILD_INFO[a3];
            for(unsigned i = 0; i < N; i++) {
                if(i < info.torsion_vals.size()) {
                    sc_bond_length[a][i] = info.bonds_vals[i];
                    sc_ctheta[a][i] = cos(M_PI-info.angles_vals[i]);
                    sc_stheta[a][i] = sin(M_PI-info.angles_vals[i]);
                    
                    auto t = info.torsion_vals[i];
                    if(t == "p") {
                        sc_type[a][i] = 1;
                    } else if(t == "i") {
                        sc_type[a][i] = 2;
                    } else {
                        float tval = stof(t);
                        sc_type[a][i] = 3;
                        sc_cchi[a][i] = cos(2*M_PI-tval);
                        sc_schi[a][i] = sin(2*M_PI-tval);
                    }
                    auto src = info.bonds_names[i];
                    src = src.substr(0,src.find('-'));
                    if(src != "CA") {
                        auto pos = find(info.atom_names.begin(), info.atom_names.end(), src);
                        auto index = distance(info.atom_names.begin(), pos);
                        sc_source_atom[a][i] = index;
                    } else {
                        sc_source_atom[a][i] = -1;    
                    }
                }
            }
        }
    }
};

const TensorBuildInfo TENSOR_BUILD_INFO;

torch::Tensor& make_backbone(torch::Tensor& M) {
    unsigned N = M.size(0);
    auto device = M.device();
//    unsigned bit_length = std::bit_width(N);  //c++20
    unsigned bit_length = 0;
    unsigned n = N;
    while(n) { n >>= 1; bit_length++;}
    
    auto indices = torch::arange(0,(int16_t)N,device);

    for(unsigned i = 0; i < bit_length; i++) {
        auto dstmask = (((1<<i)&indices) != 0).to(torch::kBool);
        auto srci = ((-1<<i)&indices)-1;
        auto src = M.index({srci.index({dstmask})});
        auto dst = M.index({dstmask});
        M.index_put_({dstmask}, torch::matmul(src,dst));
    }
    return M;
}

// Make transformation matrices given angles/lens.  Uses device of ctheta.
torch::Tensor makeTmats(long N, const torch::Tensor& ctheta, 
                    const torch::Tensor& stheta, 
                    const torch::Tensor& cchi, 
                    const torch::Tensor& schi, 
                    const torch::Tensor& lens) {
    using namespace torch::indexing;
    torch::Tensor mats = torch::zeros({N,4,4},ctheta.device());
    
    mats.index_put_({Slice(),0,0}, ctheta);
    mats.index_put_({Slice(),0,1}, -stheta);
    mats.index_put_({Slice(),0,3}, ctheta*lens);
    
    mats.index_put_({Slice(),1,0}, cchi*stheta);
    mats.index_put_({Slice(),1,1}, ctheta*cchi);
    mats.index_put_({Slice(),1,2}, schi);
    mats.index_put_({Slice(),1,3}, cchi*lens*stheta);

    mats.index_put_({Slice(),2,0}, -stheta*schi);
    mats.index_put_({Slice(),2,1}, -ctheta*schi);
    mats.index_put_({Slice(),2,2}, cchi);
    mats.index_put_({Slice(),2,3}, -lens*stheta*schi);
    
    mats.index_put_({Slice(),3,3}, 1.0);

    return mats;
}

//create coordinates for sequence seq using angles angles
//device is taken from angles
torch::Tensor make_coords(const std::string& seq, torch::Tensor& angles) {
    using namespace torch::indexing;
    int L = (int)seq.length();
    auto device = angles.device();
    
    auto fangles = angles.to(torch::kFloat32);
    
    torch::Tensor ang = torch::zeros({L+1,SC_ANGLES_START_POS},device);
    ang.index_put_({Slice(1)}, fangles.index({Slice(None,L), Slice(None,SC_ANGLES_START_POS)}));    
    ang.index_put_({Slice(),Slice(3)}, M_PI - ang.index({Slice(),Slice(3)})); // theta
    ang.index_put_({Slice(),Slice(None,3)}, 2.0*M_PI - ang.index({Slice(),Slice(None,3)})); //chi
    
    auto sins = torch::sin(ang);
    auto coss = torch::cos(ang);
    
    auto schi = sins.index({Slice(),Slice(None,3)}).flatten().index({Slice(1,-1)});
    auto cchi = coss.index({Slice(),Slice(None,3)}).flatten().index({Slice(1,-1)});
    
    auto stheta = sins.index({Slice(),Slice(3)}).flatten().index({Slice(1,-2)});
    auto ctheta = coss.index({Slice(),Slice(3)}).flatten().index({Slice(1,-2)});

    //O needs one additional angle
    auto oschi = schi.index({Slice(3,None,3)});
    auto occhi = cchi.index({Slice(3,None,3)});
    
    schi = schi.index({Slice(None,-1)});
    cchi = cchi.index({Slice(None,-1)});
                
    auto lens = BB_BUILD_INFO.ALL_BB_LENS.clone().to(device).repeat(L);
    lens[0] = 0;
    lens = lens.to(device);

    auto ncacM = makeTmats(L*3, ctheta, stheta, cchi, schi, lens);
    ncacM = make_backbone(ncacM);

    // =O
    auto t = torch::tensor(M_PI-BB_BUILD_INFO.BONDANGS.at("ca-c-o"),device);
    auto ct = torch::cos(t);
    auto st = torch::sin(t);
    auto Ol = torch::tensor(BB_BUILD_INFO.BONDLENS.at("c-o"),device);
    auto Omats = makeTmats(L, ct, st, -occhi, -oschi, Ol);

    auto oM = torch::matmul(ncacM.index({Slice(2,None,3)}),Omats);

    // get bb coords from matrices
    auto vec = torch::tensor({0.0,0.0,0.0,1.0},device);
    auto ncac = torch::matmul(ncacM,vec).index({Slice(),Slice(None,3)}).reshape({L,3,3});
    auto ocoords = torch::matmul(oM,vec).index({Slice(),Slice(None,3)}).reshape({L,1,3});
    
    //now sidechains
    auto sccoords = torch::zeros({L,NUM_COORDS_PER_RES-4,3},device);

    //all sc angles are dihedrals
    ang = 2.0*M_PI-fangles.index({Slice(),Slice(SC_ANGLES_START_POS)});
    sins = torch::sin(ang);
    coss = torch::cos(ang);
    
    //convert sequence to aa index
    vector<int> aai; aai.reserve(L);
    for(char c : seq) {
        aai.push_back(aa2i[c]);
    }
    auto seq_aa_index = torch::tensor(aai,device);

    //select out angles/bonds/etc for full seq
    auto bond_lengths = TENSOR_BUILD_INFO.sc_bond_length.index({seq_aa_index}).to(device);
    auto source_atoms = TENSOR_BUILD_INFO.sc_source_atom.index({seq_aa_index}).to(device);
    auto cthetas =  TENSOR_BUILD_INFO.sc_ctheta.index({seq_aa_index}).to(device);
    auto sthetas =  TENSOR_BUILD_INFO.sc_stheta.index({seq_aa_index}).to(device);
    auto cchis =  TENSOR_BUILD_INFO.sc_cchi.index({seq_aa_index}).to(device);
    auto schis =  TENSOR_BUILD_INFO.sc_schi.index({seq_aa_index}).to(device);
    auto types = TENSOR_BUILD_INFO.sc_type.index({seq_aa_index}).to(device);
    auto sources = TENSOR_BUILD_INFO.sc_source_atom.index({seq_aa_index}).to(device);
    
    //first iteration: build from CA
    auto seqmask = types.index({Slice(),0}).to(torch::kBool);

    lens = bond_lengths.index({seqmask,0});
    unsigned N = lens.size(0);

    ctheta = cthetas.index({seqmask,0});
    stheta = sthetas.index({seqmask,0});
    cchi = coss.index({seqmask,0});
    schi = sins.index({seqmask,0});

    auto mats = makeTmats(N, ctheta, stheta, cchi, schi, lens);
    auto prevmats = torch::matmul(ncacM.index({Slice(1,None,3)}).index({seqmask}),mats);
    auto c = torch::matmul(prevmats,vec);
    sccoords.index_put_({seqmask,0}, c.index({Slice(),Slice(None,3)}));      
    
    vector<torch::Tensor> matrices; matrices.reserve(10);
    vector<torch::Tensor> masks; masks.reserve(10);
    torch::Tensor prevmask;
    
    for(int i = 1; i < 10; i++) {
        matrices.push_back(prevmats);
        masks.push_back(seqmask);
        prevmask = seqmask;
        seqmask = types.index({Slice(),i}).to(at::kBool);
        lens = bond_lengths.index({seqmask,i});
        N = lens.size(0);
        if(N == 0) break;
        
        //theta always constant
        auto ctheta = cthetas.index({seqmask,i});
        auto stheta = sthetas.index({seqmask,i});
        
        //chi constant...
        cchi = cchis.index({seqmask,i});
        schi = schis.index({seqmask,i});
        //...unless it isn't
        auto itypes = types.index({Slice(),i});
        auto regmask = itypes == 1;
        
        if(regmask.any().item().toBool()) {
            auto regseq = regmask.index({seqmask});
            cchi.index_put_({regseq}, coss.index({regmask,i}));
            schi.index_put_({regseq}, sins.index({regmask,i}));
        }
        
        auto invmask = itypes == 2;
        if(invmask.any().item().toBool()) {
            auto invseq = invmask.index({seqmask});
            cchi.index_put_({invseq}, -coss.index({invmask,i-1}));
            schi.index_put_({invseq}, -sins.index({invmask,i-1}));
        }
        
        //check if any atoms need matrix from farther back
        if ((sources.index({Slice(),i}) != i-1).any().item().toBool()) {
            prevmats = prevmats.clone();
            
            for(int k = i-2; k >= 0; k--) {
                auto srcmask = (sources.index({Slice(),i}) == k) & seqmask;
                if(srcmask.any().item().toBool()) { //change prevmat to earlier matrix
                    prevmats.index_put_({srcmask.index({prevmask})}, matrices[k].index({srcmask.index({masks[k]})}));
                }
            }       
        }
        
        mats = makeTmats(N, ctheta, stheta, cchi, schi, lens);
        prevmats = torch::matmul(prevmats.index({seqmask.index({prevmask})}),mats);
        
        c = torch::matmul(prevmats,vec);
        sccoords.index_put_({seqmask,i}, c.index({Slice(),Slice(None,3)}));             
    }
    
    return torch::cat({ncac,ocoords,sccoords},1);
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     
    py::class_<BackboneBuildInfo>(m, "BackboneBuildInfo",py::module_local())
        .def_readonly("BONDLENS", &BackboneBuildInfo::BONDLENS)
        .def_readonly("BONDANGS", &BackboneBuildInfo::BONDANGS)
        .def_readonly("BONDTORSIONS", &BackboneBuildInfo::BONDTORSIONS);
        
    py::class_<SCResidueInfo>(m, "SCResidueInfo",py::module_local())
        .def_readonly("angles_names", &SCResidueInfo::angles_names)
        .def_readonly("angles_types", &SCResidueInfo::angles_types)
        .def_readonly("angles_vals", &SCResidueInfo::angles_vals)
        .def_readonly("atom_names", &SCResidueInfo::atom_names)
        .def_readonly("bonds_names", &SCResidueInfo::bonds_names)
        .def_readonly("bonds_types", &SCResidueInfo::bonds_types)
        .def_readonly("bonds_vals", &SCResidueInfo::bonds_vals)
        .def_readonly("torsion_names", &SCResidueInfo::torsion_names)
        .def_readonly("torsion_types", &SCResidueInfo::torsion_types)
        .def_readonly("torsion_vals", &SCResidueInfo::torsion_vals);

    
    m.def("make_backbone", &make_backbone, "make_backbone"); 
    m.def("makeTmats", &makeTmats, "makeTmats");
    m.def("make_coords", &make_coords, "Create coordinates from sequence and angles.");
    m.attr("NUM_COORDS_PER_RES") = py::int_(NUM_COORDS_PER_RES);
    m.attr("SC_ANGLES_START_POS") = py::int_(SC_ANGLES_START_POS);
    m.attr("NUM_ANGLES") = py::int_(NUM_ANGLES);
    m.attr("BB_BUILD_INFO") = BB_BUILD_INFO; 
    m.attr("SC_BUILD_INFO") = SC_BUILD_INFO;
}
