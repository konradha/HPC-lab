from pde_miniapp_py import viz

# initial
#viz.draw("simulation", 0)
#viz.plt.show()

# final
viz.draw("simulation", 100)
viz.plt.savefig("output_py.png", dpi=400)
#viz.plt.show()

