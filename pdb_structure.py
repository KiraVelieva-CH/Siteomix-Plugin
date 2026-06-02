
from io import TextIOWrapper  # necessary to check for file objects
from logging import getLogger

from .radii import charges, radii

logger = getLogger(__name__)


class PDBAtom:
    """
    Class for a single atom in a (protein) structure.
    """

    def __init__(self, line=None):
        # extracts information from a line of PDB formatted text,
        # no error checking is performed.
        self.het = False
        self.atom_number = 0
        self.name = " X "
        self.alternate = ""
        self.residue = "UNK"
        self.chain = "X"
        self.residue_number = 0
        self.insertion = " "
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.occupancy = 0.0
        self.bfactor = 0.0

        self.segi = " "
        self.element = " "
        self.crg = "  "

        # other possible atom parameters
        self.keep_old_bfactor = self.bfactor
        self.HP = 0.0
        self.radius = 2.0
        self.charge = 0.0
        self.formal_charge = 0.0
        self.par1 = 0.0
        self.par2 = 0.0
        self.pseudo_id = 0

        # property dictionary: {"prop1": value,"prop2": value"}
        self.prop_dic = {}

        if line is not None:  # read data from a PDB ATOM or HETATM line
            self.add_line(line)

    def add_line(self, line):
        """
        Read atom data from a line of PDB-formatted text,
        assumes line to start with 'ATOM' or 'HETATM'
        """
        self.het = line[:6] == "HETATM"
        self.atom_number = int(line[6:11])
        self.name = line[12:16]
        self.alternate = line[16]
        self.residue = line[17:20]
        self.chain = line[21]
        self.residue_number = line[22:26]
        self.insertion = line[26]
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])

        # in some PDB files, everything beyond the coordinates is missing
        try:
            self.occupancy = float(line[54:60])
        except ValueError:
            self.occupancy = 1.0

        try:
            self.bfactor = float(line[60:66])
        except ValueError:
            self.bfactor = 20.0

        self.segi = line[71:76]
        self.element = line[76:78]
        self.crg = line[78:80]
        self.keep_old_bfactor = self.bfactor

    def set_prop(self, prop, value=None):
        """
        Set property 'prop' of atom to value.
        """
        if hasattr(self, prop):
            setattr(self, prop, value)

    def add_prop_dic(self, prop, value=None):
        """
        Add or update custom property dictionary.
        """
        self.prop_dic.update({prop: value})

    def _print_prop_info(self):
        u = []
        u.extend(
            [
                "keep_old_bfactor",
                "radius          ",
                "charge          ",
                "formal_charge   ",
                "par1            ",
                "par2            ",
                "HP              ",
                "pseudo_id       ",
            ]
        )
        u.extend([key for key in self.prop_dic.keys()])
        u = ["%-16s" % (str(i)) for i in u]
        print(" ".join(u))

    def print_prop(self, form="table"):
        """
        Output non-standard properties in various formats
        form: 'table'
              'catalophore'
              'csv'
        """
        if form == "table":
            print(self.__str__())
            print("Properties:")
            print("-----------")
            print(".keep_old_bfactor:   %s" % self.keep_old_bfactor)
            print(".radius          :   %s" % self.radius)
            print(".charge          :   %s" % self.charge)
            print(".formal_charge   :   %s" % self.formal_charge)
            print(".par1            :   %s" % self.par1)
            print(".par2            :   %s" % self.par2)
            print(".HP              :   %s" % self.HP)
            print(".pseudo_id       :   %s" % self.pseudo_id)
            print("-------------------------")
            print("Additional: .prop_dic")
            for key, value in self.prop_dic.items():
                print("%s: %s" % (key, value))

        elif form == "catalophore":
            print(
                ",".join(
                    [
                        str(a)
                        for a in [
                            self.atom_number,
                            self.residue_number,
                            self.x,
                            self.y,
                            self.z,
                            self.keep_old_bfactor,
                            self.HP,
                        ]
                    ]
                )
            )

        else:
            u = [
                int(self.atom_number),  # point_id
                int(self.residue_number),  # cav_id
                float(self.x),  # x
                float(self.y),  # y
                float(self.z),  # z
            ]
            u.extend(
                [
                    self.keep_old_bfactor,
                    self.radius,
                    self.charge,
                    self.formal_charge,
                    self.par1,
                    self.par2,
                    self.HP,
                    self.pseudo_id,
                ]
            )
            u.extend([value for value in self.prop_dic.values()])
            u = ["%s" % (str(i)) for i in u]
            print(",".join(u))

    def __str__(self):
        return self.get_pdbstr()

    def get_pdbstr(self, prop=None, prop_key=None):

        if self.het:
            start = "HETATM"
        else:
            start = "ATOM  "

        if prop is None or prop == "bfactor":  # use the value im atom.bfactor
            value = self.bfactor
        elif prop == "prop_dic":  # use a value from 'prop_dic'
            value = self.prop_dic.get(prop_key, -1.0)
        else:  # use another property
            if hasattr(self, prop):
                value = getattr(self, prop)
            else:
                value = -1.0

        return (
            "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%2s"
            % (
                start,
                int(self.atom_number),
                str(self.name),
                str(self.alternate),
                str(self.residue),
                str(self.chain),
                int(self.residue_number),
                str(self.insertion),
                float(self.x),
                float(self.y),
                float(self.z),
                float(self.occupancy),
                value,  # float(self.bfactor),  replacement value for the B-factor
                str(self.segi),
                str(self.element),
                str(self.crg),
            )
        )


class PDBStructure:
    """
    Class for a protein structure which is read from a file-like,
    PDB-formatted object.
    """

    def __init__(self, file_obj=None):
        """
        reads a PDB file line per line. Non-atom lines are stored in a
        separate list.

        file_obj can be a file object or a list (tuple) of PDB lines
        """
        self.remarks = []  # list that contains non-atom lines
        self.atom = []  # list of instances of PdbAtom
        self.water = []  # water coordinates

        # The following attributes are currently not needed
        # self.deleted_atom = []      # as atom[], but for deleted atoms (H, HETATM,...)
        # self.dummy = []             # dummy coordinates

        # # pseudo center atoms (CavBase-like pseudo atoms)
        # self.pseudo_aliphatic = []  # aliphatic pseudo atoms
        # self.pseudo_pi = []         # pi pseudo atoms
        # self.pseudo_donor = []      # donor atoms
        # self.pseudo_acceptor = []   # acceptor atoms
        # self.pseudo_DON_ACC = []    # donor_acceptor atoms

        if file_obj is not None:
            if (
                isinstance(file_obj, list)
                or isinstance(file_obj, tuple)
                or isinstance(file_obj, TextIOWrapper)
            ):  # 'file'
                tmp = file_obj
            elif isinstance(file_obj, str):
                tmp = file_obj.split("\n")  # assuming PyMOL pdbstr
            else:
                return

            for line in tmp:
                if self._is_atom(line):
                    self.atom.append(PDBAtom(line))
                else:
                    self.remarks.append(line)

            self.add_water()

    @staticmethod
    def _is_atom(line):
        """Checks whether the line contains atom data."""
        return line[:4] == "ATOM" or line[:6] == "HETATM"

    def __str__(self):
        return self.get_pdbstr()

    def get_pdbstr(self, prop=None, prop_key=None):
        """
        Return pdbstr of atoms
        """
        return (
            "\n".join([i.get_pdbstr(prop=prop, prop_key=prop_key) for i in self.atom])
            + "\n"
        )

    def add_water(self, atoms=None, resn_list=("H2O", "HOH", "WAT")):
        """
        Put water atoms to separate atoms list called water.
        atoms: list of atom objects
        resn_list: ("H2O","HOH","WAT") residue names for water molecules.
        """
        if atoms is None:
            atoms = self.atom

        for atom in atoms:
            if atom.residue in resn_list:
                self.water.append(atom)

    def change_bfac(
        self, atom_list=None, from_prop="par1", prop_key=None, to_prop="bfactor"
    ):
        """
        Change the bfac field from old property to new property.
        atomlist: list of PdbAtom objects (standard: self.atom)

        from_prop: "par1",
                   "par2",
                   "radius"
                   "charge"
                   "occupancy"
                   ...         other properties have to be added
                   "prop_dic" additional property dictionary accessible by
                              "prop_key" value


                   "restore"   restore old b-factor value
        to_prop:   "bfactor"   currently only b-factor can be switched
                               old b-factor is still in "keep_old_bfactor"
                               (KG: This is obviously not true, anymore. See code!)
        """
        if atom_list is None:
            atom_list = self.atom
        elif hasattr(
            self, atom_list
        ):  # in case the PDBStructure object has an attribute 'atom_list'
            atom_list = getattr(self, atom_list)

        if len(atom_list) == 0:
            logger.warning("No atoms in selection.")
            return

        if from_prop == "prop_dic":  # read from property dictionary
            if prop_key:
                if atom_list[0].prop_dic.get(prop_key) is not None:
                    if hasattr(atom_list[0], to_prop):  # check atom[0]
                        for atom in atom_list:
                            setattr(atom, to_prop, atom.prop_dic.get(prop_key))
                    else:
                        logger.warning("No %s property found in atom.", str(to_prop))
                else:
                    logger.warning(
                        "No %s property found in atom prop_dic.", str(prop_key)
                    )
                    return

        elif from_prop == "restore":  # restore original B-factor
            for atom in atom_list:
                atom.bfactor = atom.keep_old_bfactor

        elif hasattr(atom_list[0], from_prop) and hasattr(  # check atom[0]
            atom_list[0], to_prop
        ):
            for atom in atom_list:
                setattr(atom, to_prop, getattr(atom, from_prop))

    def assign_rad_hp_chg(
        self, HP_dict=None, HP_column=2, rad_dict=None, rad_column=0, chg_dict=None
    ):
        """
        Assign radii and hydrophobicity values to atoms in a structure
        :param chg_dict:
        :param HP_dict:
        :param HP_column:
        :param rad_dict:
        :param rad_column:
        :return:
        """
        if rad_dict is None:
            rad_dict = radii  # internal radius dictionary
        if HP_dict is None:
            HP_dict = radii  # internal radius/HP dictionary
        if chg_dict is None:
            chg_dict = charges

        for atom in self.atom:
            queries = [  # atom queries with decreasing specificity
                (atom.residue, atom.name),
                (None, atom.name),
                (None, atom.name[:3]),
                (None, atom.name[:2]),
            ]

            # set atom radii
            for query in queries:
                value = rad_dict.get(query)
                if value:
                    atom.radius = value[rad_column]
                    break
            else:
                logger.warning(
                    "ATOM %s %s %s unknown. Radius is set to %.1f"
                    % (atom.atom_number, atom.residue, atom.name, 2.0)
                )
                atom.radius = 2.0

            # set HC values
            for query in queries:
                value = HP_dict.get(query)
                if value:
                    atom.HP = value[HP_column]
                    break
            else:
                logger.warning(
                    "ATOM %s %s %s unknown. HC value is set to %.1f"
                    % (atom.atom_number, atom.residue, atom.name, 0.0)
                )
                atom.HP = 0.0

            # set partial charges
            for query in queries:
                value = chg_dict.get(query)
                if value:
                    atom.charge = value
                    break
            else:
                atom.charge = 0.0
