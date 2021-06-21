from Lego import Lego
from scipy.spatial import distance


def get_key(dict_lego, lego, minDist):

    #dict_values = list(dict_lego.values())

    for key, values in dict_lego.items():
        if lego.color in values[:][0] and distance.euclidean(lego.ref_point, values[:][1]) <= minDist:
            return key

def update_dict(dict_lego, lego, minDist):
    key = get_key(dict_lego, lego, minDist)
    if key != None:
        """ update lego's reference point """
        dict_lego[key] = [lego.color, lego.ref_point]

    else:
        """ create new entry in dict """
        dict_lego[len(dict_lego) + 1] = [lego.color, lego.ref_point]


lego1 = Lego("orange", ref_point=(200, 400))
lego2 = Lego("blue", ref_point=(200, 600))

lego3 = Lego("red", ref_point=(200, 800))
lego4 = Lego("orange", ref_point=(300, 600))
lego5 = Lego("blue", ref_point=(200, 700))
#lego2 = Lego("red", (100, 200), "2:1", None, rect_non_rect = True)
#lego3 = Lego("blue", (100, 200), "2:1", None, rect_non_rect = False)

"""
print("lego1")
lego1.print_lego()
print("\nlego2")
lego2.print_lego()
print("\nlego3")
lego3.print_lego()
"""
dict_lego = {1 : (lego1.color, lego1.ref_point), 2: (lego2.color, lego2.ref_point)}

lego_lst = [lego3, lego4, lego5]

print("before update")
for key, value in dict_lego.items():
    print(key, ' : ', value)

for lego in lego_lst:
    update_dict(dict_lego, lego, 400)

print("after update")
for key, value in dict_lego.items():
    print(key, ' : ', value)

