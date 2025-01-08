
from enum import Enum
from functools import lru_cache


class Databases(Enum):
    IITD_database = 'IITD_database'
    UTIRIS_database = 'utiris eye database/RGB Images'
    DISEASES = 'diseases'


class EyeDiseases(Enum):
    HealtyEye = 'Zdravo oko'
    Cataract = 'Siva mrena'
    # ArcusSenilis = 'Arcus senilis'
    Mydriasis = 'Midriaza'
    Miosis = 'Mioza'
    IritisOrKeratitisPupilShape = 'Vnetje sarenice ali rozenice'
    # Jaundice = 'Zlatenica'
    # PrevelikaSarenica = 'Prevelika sarenica'
    # PremajhnaSarenica = 'Premajhna sarenica'
    # KayserFleischerjevRing = 'Kayser fleischerjev obroc'
    # NoIris = 'no iris'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def str_to_object(string):
        for s in EyeDiseases:
            if string.lower().strip() == s.value.lower().strip():
                return EyeDiseases[s.name]

    @staticmethod
    def list_to_objects(string):
        string = string.replace('{', '').replace('}', '').split(',')
        return list(filter(lambda a: a is not None, [EyeDiseases.str_to_object(s) for s in string]))

    @lru_cache(1)
    def diseases_table():
        return list(map(str, EyeDiseases))

    def exist(attr):
        try:
            EyeDiseases[attr]
        except:
            return False
        return True
