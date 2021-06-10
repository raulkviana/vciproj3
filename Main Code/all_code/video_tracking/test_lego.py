from lego import Lego

lego1 = Lego("yellow")
lego2 = Lego("red", (100, 200), "2:1", None, rect_non_rect = True)
lego3 = Lego("blue", (100, 200), "2:1", None, rect_non_rect = False)

print("lego1")
lego1.print_lego()
print("\nlego2")
lego2.print_lego()
print("\nlego3")
lego3.print_lego()
